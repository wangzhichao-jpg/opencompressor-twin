"""Retrospective fixed-budget learning curves; selectors see INPUTS ONLY.

This compares random and maximin space-filling order, not a completed adaptive
information-gain strategy. Calibration/final-test outputs cannot select points,
choose a stopping budget or tune any selector. Repeated curves are secondary
analysis on a published held-out set, not new independent blind experiments.
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .baselines import evaluate_ahri10
from .core import TARGETS, OperatingPoint, load_dataset
from .properties import suction_properties
from .reporting import curve_report, write_csv
from .workflow import (
    build_artifact,
    check_split,
    domain,
    environment,
    linear_baseline,
    metrics,
    predict_point,
    vector,
    write_json,
)


def input_selection_order(group_points, initial_groups, strategy, seed):
    if strategy not in {"random", "maximin"}:
        raise ValueError("Selection strategy must be random or maximin")
    if (
        not initial_groups
        or len(initial_groups) != len(set(initial_groups))
        or not set(initial_groups).issubset(group_points)
    ):
        raise ValueError("Unique nonempty initial groups must be a subset of the training pool")
    if any(not points or any(not isinstance(p, OperatingPoint) for p in points) for points in group_points.values()):
        raise ValueError("Selector accepts OperatingPoint inputs only, never Observation targets")
    # Representatives are for input coverage only, not fabricated measured points.
    centers = {g: np.mean([vector(p) for p in points], axis=0) for g, points in group_points.items()}
    order = list(initial_groups)
    remaining = sorted(set(group_points) - set(order))
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    if strategy == "random":
        return order + remaining
    scale = np.ptp(list(centers.values()), axis=0)
    scale[scale == 0] = 1
    while remaining:
        reference = np.array([centers[g] for g in order])
        scores = [float(np.linalg.norm((reference - centers[g]) / scale, axis=1).min()) for g in remaining]
        order.append(remaining.pop(int(np.argmax(scores))))
    return order


def evaluate_fixed_orders(data, pool, test, orders, budgets, include_residual=False):
    """Score already-frozen orders; no evaluator feedback reaches selectors."""
    rows, failures = [], []
    families = ("semiempirical", "affine", "ahri10") + (("semiempirical_rbf",) if include_residual else ())
    # Reuse identical subsets across seeds/strategies without changing information.
    cache = {}
    for order in orders:
        for budget in sorted(budgets):
            if len(order["groups"]) < budget:
                for family in families:
                    row = {
                        "strategy": order["strategy"],
                        "seed": order["seed"],
                        "budget_groups": budget,
                        "family": family,
                        "status": "unavailable",
                        "test_total": len(test),
                        "reason": "Selector stopped before requested budget; no silent fallback/reveal",
                    }
                    rows.append(row)
                    failures.append(row.copy())
                continue
            selected = set(order["groups"][:budget])
            key = tuple(sorted(selected))
            if key not in cache:
                observed = [o for o in pool if o.group_id in selected]
                domain_artifact = {"training_inputs": [asdict(o.point) for o in observed]}
                common_domain = [{"point_id": o.point.point_id, **domain(domain_artifact, o.point)} for o in test]
                try:
                    artifact = build_artifact(replace(data, observations=observed), observed)
                    predicted = [predict_point(artifact, o.point) for o in test]
                    semi = {"status": "available", "metrics": metrics(test, predicted)}
                except ValueError as exc:
                    semi = {"status": "failed", "reason": str(exc)}
                affine = linear_baseline(observed, test, common_domain)
                ahri = evaluate_ahri10(data.spec, observed, test, common_domain)
                cache[key] = {"semiempirical": semi, "affine": affine, "ahri10": ahri}
                if include_residual:
                    try:
                        if semi["status"] != "available":
                            raise ValueError("Base fit unavailable for residual extension")
                        from .discrepancy import add_residual_correction

                        corrected = add_residual_correction(artifact, observed)
                        predictions = [predict_point(corrected, o.point) for o in test]
                        cache[key]["semiempirical_rbf"] = {"status": "available", "metrics": metrics(test, predictions)}
                    except ValueError as exc:
                        cache[key]["semiempirical_rbf"] = {"status": "failed", "reason": str(exc)}
            for family, score in cache[key].items():
                row = {
                    "strategy": order["strategy"],
                    "seed": order["seed"],
                    "budget_groups": budget,
                    "family": family,
                    "status": score["status"],
                    "test_total": len(test),
                    "selected_groups_sha256": hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                }
                if score["status"] == "available":
                    m = score["metrics"]
                    row.update(predicted=m["predicted"], refused_or_failed=m["refused_or_failed"])
                    for target in TARGETS:
                        for name in ("mare_percent", "p95_are_percent", "max_are_percent"):
                            row[target + "_" + name] = m[target][name] if m[target] is not None else None
                else:
                    row["reason"] = score.get("reason", "unavailable")
                    failures.append(row.copy())
                rows.append(row)
    summary = []
    for strategy in dict.fromkeys(o["strategy"] for o in orders):
        for family in families:
            for budget in sorted(budgets):
                batch = [
                    r
                    for r in rows
                    if r["strategy"] == strategy and r["family"] == family and r["budget_groups"] == budget
                ]
                accepted = [r for r in batch if r["status"] == "available"]
                item = {
                    "strategy": strategy,
                    "family": family,
                    "budget_groups": budget,
                    "trials": len(batch),
                    "available_trials": len(accepted),
                    "mean_predicted_points": float(np.mean([r["predicted"] for r in accepted])) if accepted else None,
                }
                for target in TARGETS:
                    values = [r[target + "_mare_percent"] for r in accepted if r[target + "_mare_percent"] is not None]
                    item[target + "_mare_mean_percent"] = float(np.mean(values)) if values else None
                    item[target + "_mare_std_percent"] = float(np.std(values)) if values else None
                summary.append(item)
    return rows, failures, summary


def run_learning_curve(csv_path, meta_path, split, output, budgets=(9, 12, 16, 24, 32, 40), seeds=tuple(range(10))):
    data = load_dataset(csv_path, meta_path)
    check_split(data.observations, split)
    initial = split.get("mandatory_hull_training_groups")
    if not isinstance(initial, list) or len(initial) < 4:
        raise ValueError(
            "Learning curve requires >=4 explicit mandatory_hull_training_groups in frozen split; no automatic outcome-based initial selection"
        )
    pool = [o for o in data.observations if o.group_id in split["train"]]
    test = [o for o in data.observations if o.group_id in split["test"]]
    for o in pool + test:
        suction_properties(o.point, data.spec.refrigerant)
    group_points = defaultdict(list)
    for o in pool:
        group_points[o.group_id].append(o.point)
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(
            isinstance(b, bool) or not isinstance(b, int) or b < len(initial) or b > len(group_points) for b in budgets
        )
    ):
        raise ValueError(
            "Budgets must be unique integers between mandatory initial count and training-pool group count"
        )
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("Seeds must be unique nonnegative integers")
    # Freeze orders/protocol before computing any test score. Selections contain no y.
    orders = [
        {"strategy": strategy, "seed": seed, "groups": input_selection_order(group_points, initial, strategy, seed)}
        for strategy in ("random", "maximin")
        for seed in seeds
    ]
    protocol = {
        "dataset_fingerprint": data.fingerprint,
        "split": split,
        "budgets": list(budgets),
        "seeds": list(seeds),
        "initial_group_count": len(initial),
        "orders": orders,
        "information_boundary": "Only training-pool inputs determine orders; only selected groups reveal outputs; final test only scored at fixed budgets",
        "evaluation_status": "secondary retrospective fixed-budget analysis; not fresh blind validation; no measured-time claim",
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing learning-curve protocol differs; use a new output/protocol, do not overwrite")
    write_json(protocol_path, protocol)
    rows, failures, summary = evaluate_fixed_orders(data, pool, test, orders, budgets)
    source_limits = data.metadata.get("known_limitations", [])
    if isinstance(source_limits, str):
        source_limits = [source_limits]
    result = {
        "kind": "learning_curve",
        "dataset_id": data.metadata["dataset_id"],
        "source_kind": data.metadata["source_kind"],
        "source": data.metadata["source"],
        "compressor": asdict(data.spec),
        "power_boundary": data.metadata["power_boundary"],
        "speed_definition": data.metadata["speed_definition"],
        "environment": environment(),
        "dataset_fingerprint": data.fingerprint,
        "protocol_sha256": protocol_hash,
        "rows": rows,
        "summary": summary,
        "failures": failures,
        "limitations": [
            "Not adaptive information-gain selection: random and input-only maximin are comparison strategies",
            "Initial hull coverage is given to every strategy; its measurement cost is INCLUDED in budget",
            "Test set has been scored before; this is secondary analysis, not a new prospective blind experiment",
            "Curves do not set online stopping rules, tune models or prove bench-time savings",
            "10 repeated orders share one machine and one final set; repeat SD is not independent-laboratory uncertainty",
            "No prediction intervals evaluated along this curve; calibration groups remain reserved and unused",
            "Budget counts revealed training groups only; reserved calibration/test groups are excluded, not total experimental-program cost",
        ]
        + source_limits,
    }
    write_json(output / "result.json", result)
    write_csv(output / "trials.csv", rows)
    write_csv(output / "summary.csv", summary)
    curve_report(result, output / "report.html")
    return result
