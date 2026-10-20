"""Development-only grouped CV and design study; never accepts original holdout.

All acquisition trajectories are frozen before any CV scoring. Dataset exports
exclude original calibration/final labels entirely. Development CV is not fresh
independent validation or a measured-cost study.
"""

import hashlib
import html
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .baselines import evaluate_ahri10
from .core import TARGETS, load_dataset
from .discrepancy import EXTENSION_VERSION, add_residual_correction
from .evaluation import evaluate_fixed_orders, input_selection_order
from .kernels import KERNEL_RIDGE, KERNEL_VERSION, LENGTH_SCALE, kernel_selection_order
from .models import MODEL_VERSION
from .replay import TrainingPoolOracle, adaptive_information_order
from .reporting import write_csv
from .workflow import build_artifact, domain, environment, linear_baseline, predict_point, write_json

FAMILIES = ("semiempirical", "affine", "ahri10", "semiempirical_rbf")
STRATEGIES = ("random", "maximin", "d_optimal", "kernel_ivr")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _freeze(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Frozen development evidence differs; choose new output directory")
    write_json(path, value)


def validate_development(data, config):
    if data.metadata.get("research_partition") != "development_only" or config.get("scope") != "development_only":
        raise ValueError(
            "Development study requires a separate development-only export; original holdout CSV is not accepted"
        )
    groups = {o.group_id for o in data.observations}
    if groups != set(config["development_groups"]) or groups.intersection(config["forbidden_groups"]):
        raise ValueError("Development groups differ or contain forbidden calibration/final groups")
    if len(groups) != len(data.observations):
        raise ValueError(
            "This CV protocol expects one aggregated observation per group; define repeat-group scoring first"
        )
    initial = config["initial_groups"]
    if len(initial) < 4 or len(initial) != len(set(initial)) or not set(initial).issubset(groups):
        raise ValueError("Invalid development initial groups")
    if (
        config["length_scale"] != LENGTH_SCALE
        or config["kernel_ridge"] != KERNEL_RIDGE
        or config["strategies"] != list(STRATEGIES)
    ):
        raise ValueError("Development candidates/hyperparameters differ from this protocol version")
    folds = config["folds"]
    held = []
    for i, fold in enumerate(folds):
        train, validation = fold["train"], fold["validation"]
        if (
            fold["fold"] != i
            or not validation
            or len(train) != len(set(train))
            or len(validation) != len(set(validation))
            or set(train).intersection(validation)
            or set(train + validation) != groups
            or not set(initial).issubset(train)
        ):
            raise ValueError("Invalid development fold partition")
        held.extend(validation)
    if len(folds) != 5 or len(held) != len(set(held)) or set(held) != groups - set(initial):
        raise ValueError("Five folds must validate each noninitial group exactly once")
    budgets, seeds = config["budgets"], config["seeds"]
    if (
        not budgets
        or len(budgets) != len(set(budgets))
        or any(
            isinstance(b, bool)
            or not isinstance(b, int)
            or not len(initial) <= b <= min(len(f["train"]) for f in folds)
            for b in budgets
        )
    ):
        raise ValueError("Invalid development budgets")
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("Invalid development seeds")
    from .properties import suction_properties

    for observed in data.observations:
        suction_properties(observed.point, data.spec.refrigerant)
    initial_domain = {"training_inputs": [asdict(o.point) for o in data.observations if o.group_id in initial]}
    if any(domain(initial_domain, o.point)["status"].startswith("refused") for o in data.observations):
        raise ValueError("Initial development hull must cover all development inputs")


def _full_fit(data, train, test):
    def failed(status, error):
        return [{"point_id": o.point.point_id, "status": status, "error": str(error)} for o in test]

    matched = [
        {"point_id": o.point.point_id, **domain({"training_inputs": [asdict(a.point) for a in train]}, o.point)}
        for o in test
    ]
    try:
        base = build_artifact(replace(data, observations=train), train)
        predictions = [predict_point(base, o.point) for o in test]
    except ValueError as exc:
        base = None
        predictions = failed("failed_base_fit", exc)
    affine = linear_baseline(train, test, matched)
    ahri = evaluate_ahri10(data.spec, train, test, matched)
    try:
        if base is None:
            raise ValueError("Base fit unavailable for residual extension")
        corrected = add_residual_correction(base, train)
        extended = [predict_point(corrected, o.point) for o in test]
    except ValueError as exc:
        extended = failed("failed_residual_fit", exc)
    return {
        "semiempirical": predictions,
        "affine": affine.get("predictions", failed("unavailable_affine", affine.get("reason", ""))),
        "ahri10": ahri.get("predictions", failed("unavailable_ahri10", ahri.get("reason", ""))),
        "semiempirical_rbf": extended,
    }


def _point_rows(test, predictions, fold):
    rows = []
    for family, items in predictions.items():
        if len(items) != len(test):
            items = [{"status": "unavailable"} for _ in test]
        for o, p in zip(test, items):
            row = {
                "family": family,
                "fold": fold,
                "group_id": o.group_id,
                **asdict(o.point),
                "status": p["status"],
                "pressure_ratio": o.point.discharge_pressure_pa / o.point.suction_pressure_pa,
            }
            if p.get("error"):
                row["reason"] = p["error"]
            for target in TARGETS:
                row[target + "_observed"] = getattr(o, target)
                if target in p:
                    row[target + "_predicted"] = p[target]
                    row[target + "_signed_error_percent"] = (p[target] / getattr(o, target) - 1) * 100
            rows.append(row)
    return rows


def _correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def diagnose_rows(rows):
    summary = []
    for family in FAMILIES:
        all_rows = [r for r in rows if r["family"] == family]
        valid = [r for r in all_rows if all(t + "_signed_error_percent" in r for t in TARGETS)]
        item = {
            "family": family,
            "groups": len(all_rows),
            "predicted": len(valid),
            "refused_or_failed": len(all_rows) - len(valid),
        }
        for target in TARGETS:
            errors = np.array([r[target + "_signed_error_percent"] for r in valid])
            if not len(errors):
                item[target] = None
                continue
            trends = {
                x: _correlation([r[x] for r in valid], errors)
                for x in ("pressure_ratio", "suction_pressure_pa", "discharge_pressure_pa")
            }
            ratio = np.log([r["pressure_ratio"] for r in valid])
            pressure = np.log([r["suction_pressure_pa"] for r in valid])
            design = np.column_stack([np.ones(len(ratio)), ratio, ratio**2])
            error_remainder = errors - design @ np.linalg.lstsq(design, errors, rcond=None)[0]
            pressure_remainder = pressure - design @ np.linalg.lstsq(design, pressure, rcond=None)[0]
            item[target] = {
                "mare_percent": float(np.mean(np.abs(errors))),
                "mean_signed_error_percent": float(np.mean(errors)),
                "p95_are_percent": float(np.percentile(np.abs(errors), 95)),
                "max_are_percent": float(np.max(np.abs(errors))),
                "input_error_correlations": trends,
                "absolute_pressure_partial_correlation_after_logPR_quadratic": _correlation(
                    pressure_remainder, error_remainder
                ),
            }
        summary.append(item)
    return summary


def summarize_trials(rows):
    summary = []
    for strategy in STRATEGIES:
        for family in FAMILIES:
            for budget in sorted({r["budget_groups"] for r in rows}):
                batch = [
                    r for r in rows if (r["strategy"], r["family"], r["budget_groups"]) == (strategy, family, budget)
                ]
                valid = [r for r in batch if r["status"] == "available" and r.get("predicted", 0) > 0]
                weight = sum(r["predicted"] for r in valid)
                item = {
                    "strategy": strategy,
                    "family": family,
                    "budget_groups": budget,
                    "trials": len(batch),
                    "available_trials": len(valid),
                    "expected_prediction_instances": sum(r["test_total"] for r in batch),
                    "accepted_prediction_instances": weight,
                }
                for target in TARGETS:
                    item[target + "_mare_percent"] = (
                        sum(r[target + "_mare_percent"] * r["predicted"] for r in valid) / weight if weight else None
                    )
                item["worst_response_mare_percent"] = (
                    max(item[t + "_mare_percent"] for t in TARGETS) if weight else None
                )
                summary.append(item)
    return summary


def run_development_study(csv_path, meta_path, config, output):
    # Reject original metadata BEFORE opening any original target CSV.
    declared = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if declared.get("research_partition") != "development_only" or config.get("scope") != "development_only":
        raise ValueError(
            "Development study requires a separate development-only export; original holdout CSV is not accepted"
        )
    data = load_dataset(csv_path, meta_path)
    validate_development(data, config)
    output = Path(output)
    protocol = {
        "config": config,
        "dataset_fingerprint": data.fingerprint,
        "model_version": MODEL_VERSION,
        "kernel_version": KERNEL_VERSION,
        "extension_version": EXTENSION_VERSION,
        "evaluation": "DEVELOPMENT ONLY; all orders frozen before CV scoring; no original calibration/final labels accepted",
    }
    _freeze(output / "protocol.json", protocol)
    initial = config["initial_groups"]
    traces = []
    for fold in config["folds"]:
        pool = [o for o in data.observations if o.group_id in fold["train"]]
        group_points = {o.group_id: [o.point] for o in pool}
        for strategy in STRATEGIES:
            for seed in config["seeds"]:
                if strategy == "d_optimal":
                    trace = adaptive_information_order(
                        data.spec,
                        data.metadata,
                        data.fingerprint,
                        TrainingPoolOracle(pool),
                        group_points,
                        initial,
                        seed,
                        max_groups=max(config["budgets"]),
                    )
                elif strategy == "kernel_ivr":
                    trace = kernel_selection_order(group_points, initial, seed, max(config["budgets"]))
                else:
                    trace = {
                        "strategy": strategy,
                        "seed": seed,
                        "groups": input_selection_order(group_points, initial, strategy, seed)[
                            : max(config["budgets"])
                        ],
                        "status": "completed",
                        "events": [],
                    }
                selected = trace["groups"]
                if (
                    len(selected) != len(set(selected))
                    or not set(selected).issubset(group_points)
                    or selected[: len(initial)] != initial
                    or (trace["status"] == "completed" and len(selected) != max(config["budgets"]))
                ):
                    raise ValueError("Development selection trace violated fold pool/horizon")
                if strategy == "d_optimal" and trace["reveal_log"] != selected:
                    raise ValueError("Development reveal ledger differs from selections")
                traces.append({**trace, "fold": fold["fold"]})
    trace_payload = {"protocol_sha256": _hash(protocol), "traces": traces}
    _freeze(output / "selection_traces.json", trace_payload)
    # This is the first point at which any development validation score is used.
    trials, point_rows = [], []
    for fold in config["folds"]:
        train = [o for o in data.observations if o.group_id in fold["train"]]
        validation = [o for o in data.observations if o.group_id in fold["validation"]]
        orders = [t for t in traces if t["fold"] == fold["fold"]]
        rows, _, _ = evaluate_fixed_orders(data, train, validation, orders, config["budgets"], include_residual=True)
        trials.extend({**r, "fold": fold["fold"]} for r in rows)
        point_rows.extend(_point_rows(validation, _full_fit(data, train, validation), fold["fold"]))
    try:
        full = build_artifact(data, data.observations)
        write_json(output / "model_development_base.json", full)
        write_json(output / "model_development_rbf.json", add_residual_correction(full, data.observations))
        write_json(output / "model_development_status.json", {"status": "available", "role": "development fit only"})
    except ValueError as exc:
        write_json(
            output / "model_development_status.json",
            {"status": "failed", "reason": str(exc), "role": "development fit only; no fallback model"},
        )
    train_rows = _point_rows(
        data.observations, _full_fit(data, data.observations, data.observations), "training_fit_NOT_validation"
    )
    failures = [r for r in trials if r["status"] != "available"]
    audit = {
        "development_groups": len(data.observations),
        "unique_oof_groups": len({r["group_id"] for r in point_rows}),
        "fold_training_group_counts": [len(f["train"]) for f in config["folds"]],
        "original_calibration_or_final_groups_in_data": len(
            set(config["forbidden_groups"]).intersection(o.group_id for o in data.observations)
        ),
        "total_orders": len(traces),
        "selector_stops": sum(t["status"] != "completed" for t in traces),
        "max_acquired_groups": max(len(t["groups"]) for t in traces),
        "adaptive_fit_attempts": sum(t.get("fit_attempts", 0) for t in traces),
        "unique_orders_per_fold": {
            str(f["fold"]): {
                s: len({_hash(t["groups"]) for t in traces if t["fold"] == f["fold"] and t["strategy"] == s})
                for s in STRATEGIES
            }
            for f in config["folds"]
        },
    }
    result = {
        "kind": "development_study",
        "evidence_label": "DEVELOPMENT CROSS-VALIDATION ONLY - NOT new independent validation",
        "dataset_id": data.metadata["dataset_id"],
        "source_kind": data.metadata["source_kind"],
        "source": data.metadata["source"],
        "compressor": asdict(data.spec),
        "power_boundary": data.metadata["power_boundary"],
        "speed_definition": data.metadata["speed_definition"],
        "license": data.metadata["license"],
        "dataset_fingerprint": data.fingerprint,
        "environment": environment(),
        "protocol_sha256": _hash(protocol),
        "trace_canonical_sha256": _hash(trace_payload),
        "audit": audit,
        "oof_metrics": diagnose_rows(point_rows),
        "training_fit_NOT_validation": diagnose_rows(train_rows),
        "oof_predictions": point_rows,
        "rows": trials,
        "failures": failures,
        "summary": summarize_trials(trials),
        "limitations": [
            "Original calibration/final24 groups absent from data and study; no old-final tuning/scoring",
            f"{audit['unique_oof_groups']} unique interior groups scored once per full-fold diagnostic, not boundary generalization",
            "Five overlapping training folds and repeated seeds are not independent physical experiments; no inferential CI",
            "Input-error/partial correlations describe development residual structure, not causality or measurement uncertainty",
            "Fixed RBF length_scale0.5/ridge0.01; no CV hyperparameter search; model-form bias can remain",
            "RBF adds two residual coefficients per observed condition group; NOT a four-parameter-only improvement",
            "Kernel variance omits base-parameter/model-form uncertainty; original intervals invalidated, no new interval calibration here",
            "Kernel IVR is input-only fixed-prior design, not empirically learned/adaptive noise uncertainty",
            "Curves compare point predictions; cost1 per acquired group; initial hull included, no total-program-time saving",
        ],
    }
    write_json(output / "result.json", result)
    write_csv(output / "trials.csv", trials)
    write_csv(output / "summary.csv", result["summary"])
    write_csv(output / "oof_residuals.csv", point_rows)
    write_csv(output / "training_residuals_NOT_validation.csv", train_rows)
    development_report(result, point_rows, output / "report.html")
    return result


def development_report(result, points, path):
    def table(rows, keys):
        def cell(value):
            return f"{value:.4f}" if isinstance(value, float) else html.escape(str(value))

        return (
            "<table><tr>"
            + "".join("<th>" + html.escape(k) + "</th>" for k in keys)
            + "</tr>"
            + "".join("<tr>" + "".join("<td>" + cell(r.get(k, "—")) + "</td>" for k in keys) + "</tr>" for r in rows)
            + "</table>"
        )

    diagnostic = []
    for row in result["oof_metrics"]:
        record = {"family": row["family"], "predicted": row["predicted"], "groups": row["groups"]}
        for t in TARGETS:
            record[t + " MARE (%)"] = row[t]["mare_percent"] if row[t] else None
        diagnostic.append(record)
    metadata = (
        "<p>Dataset: "
        + html.escape(result["dataset_id"])
        + "<br>Compressor: "
        + html.escape(str(result["compressor"]))
        + "<br>Power boundary: "
        + html.escape(result["power_boundary"])
        + "<br>Speed basis: "
        + html.escape(result["speed_definition"])
        + " (nominal-equivalent is NOT per-point shaft measurement)<br>Source: "
        + html.escape(result["source"])
        + "</p>"
    )
    body = (
        metadata
        + "<h2>Full-fold OOF: "
        + html.escape(str(sorted(set(result["audit"]["fold_training_group_counts"]))))
        + " training groups; "
        + str(result["audit"]["unique_oof_groups"])
        + " unique interior validation groups</h2>"
        + table(diagnostic, list(diagnostic[0]))
    )
    palette = {"semiempirical": "#167394", "affine": "#4e9c4b", "ahri10": "#ce7b08", "semiempirical_rbf": "#8a4ba4"}
    for target in TARGETS:
        valid = [r for r in points if target + "_signed_error_percent" in r]
        maxerr = max((abs(r[target + "_signed_error_percent"]) for r in valid), default=1) * 1.1 or 1
        low = min((r["suction_pressure_pa"] for r in valid), default=0)
        high = max((r["suction_pressure_pa"] for r in valid), default=1)
        circles = "".join(
            f'<circle cx="{55 + 500 * (r["suction_pressure_pa"] - low) / max(high - low, 1):.2f}" cy="{150 - 100 * r[target + "_signed_error_percent"] / maxerr:.2f}" r="3" fill="{palette[r["family"]]}"><title>{html.escape(r["family"] + " " + r["point_id"])}</title></circle>'
            for r in valid
        )
        body += f'<h3>{target}: OOF signed error vs suction pressure</h3><svg viewBox="0 0 640 300"><path d="M55 45V250H570M55 150H570" fill="none" stroke="#aaa"/>{circles}<text x="55" y="280">{low / 1e6:.3f} to {high / 1e6:.3f} MPa; signed error +/-{maxerr:.2f}%</text></svg>'
    body += (
        "<p>"
        + html.escape(" · ".join(palette))
        + "</p><p>Colors: blue semiempirical; green affine; orange AHRI10; purple semiempirical+RBF.</p>"
    )
    body += "<h2>Budget comparisons: all strategies, no cherry-picked winner</h2><p>MARE is weighted by accepted CV prediction instances. Check availability; missing trials are retained. Repeated seeds do not add independent data.</p>"
    for family in FAMILIES:
        body += (
            "<h3>"
            + html.escape(family)
            + "</h3>"
            + table(
                [r for r in result["summary"] if r["family"] == family],
                [
                    "strategy",
                    "budget_groups",
                    "available_trials",
                    "trials",
                    TARGETS[0] + "_mare_percent",
                    TARGETS[1] + "_mare_percent",
                    "worst_response_mare_percent",
                ],
            )
        )
    body += (
        "<h2>Audit / limitations / residual trends</h2><pre>"
        + html.escape(
            json.dumps(
                {"audit": result["audit"], "limitations": result["limitations"], "oof_metrics": result["oof_metrics"]},
                indent=2,
            )
        )
        + "</pre>"
    )
    evidence = result["evidence_label"]
    if result.get("source_kind") == "synthetic":
        evidence = "SYNTHETIC SOFTWARE SELF-TEST ONLY · " + evidence
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenCompressor-Twin development research</title><style>body{font:15px Arial;background:#f5f8fa;max-width:1180px;margin:30px auto;color:#213c50}h1,h2{color:#09627c}.warn{padding:18px;background:#fff0cf;border-left:5px solid #be7900}table{border-collapse:collapse;background:white;width:100%;margin:15px 0}th,td{padding:9px;border:1px solid #d5e0e5;text-align:right}th{font-size:12px;overflow-wrap:anywhere}td:first-child{text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px}svg{background:white;max-width:640px;width:100%}</style><h1>OpenCompressor-Twin · development research</h1><p class="warn">'
        + html.escape(evidence)
        + "</p><p>Protocol SHA256: "
        + html.escape(result["protocol_sha256"])
        + "</p>"
        + body
        + "</html>"
    )
    Path(path).write_text(page, encoding="utf-8")
