"""Single scoring run of the frozen round-5 external protocol. Refuses reruns.

Order: verify protocol + evaluation code hashes -> freeze all selection traces ->
score primary comparison -> score secondary design curves -> apply predeclared criteria.
"""

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octwin.core import TARGETS, load_dataset  # noqa: E402
from octwin.discrepancy import add_residual_correction  # noqa: E402
from octwin.evaluation import evaluate_fixed_orders, input_selection_order  # noqa: E402
from octwin.kernels import kernel_selection_order  # noqa: E402
from octwin.replay import TrainingPoolOracle, adaptive_information_order  # noqa: E402
from octwin.reporting import write_csv  # noqa: E402
from octwin.workflow import benchmark, build_artifact, calibrate_interval, metrics, predict_point, write_json  # noqa: E402

OUT = ROOT / "runs/round5_external"
DATASETS = ["arep11_r410a_65f", "arep11_r32_65f"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_freeze():
    freeze = json.loads((ROOT / "data_sources/round5_protocol_freeze.json").read_text(encoding="utf-8"))
    for f, h in freeze["protocol_files"].items():
        if sha(ROOT / f) != h:
            raise SystemExit(f"Protocol file changed after freeze: {f}")
    for f, h in freeze["evaluation_code_sha256"].items():
        if sha(ROOT / f) != h:
            raise SystemExit(f"Evaluation code differs from frozen dev3 code: {f}")
    return freeze


def max_mare(m):
    if m is None or m.get("predicted", 0) == 0 or any(m.get(t) is None for t in TARGETS):
        return None
    return max(m[t]["mare_percent"] for t in TARGETS)


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("Round-5 scoring already performed; single-run protocol forbids rerun")
    freeze = verify_freeze()
    data, splits, pools, tests, budgets = {}, {}, {}, {}, {}
    traces = []
    # 1) Freeze every selection trace before any scoring.
    for name in DATASETS:
        folder = ROOT / "data/public" / name
        d = load_dataset(folder / "data.csv", folder / "metadata.json")
        s = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        pool = [o for o in d.observations if o.group_id in s["train"]]
        data[name], splits[name], pools[name] = d, s, pool
        tests[name] = [o for o in d.observations if o.group_id in s["test"]]
        initial = s["mandatory_hull_training_groups"]
        group_points = {o.group_id: [o.point] for o in pool}
        budgets[name] = sorted({b for b in (len(initial), 12, 16, 24, len(pool)) if len(initial) <= b <= len(pool)})
        horizon = max(budgets[name])
        for strategy in ("random", "maximin", "d_optimal", "kernel_ivr"):
            for seed in range(5):
                if strategy == "d_optimal":
                    t = adaptive_information_order(
                        d.spec,
                        d.metadata,
                        d.fingerprint,
                        TrainingPoolOracle(pool),
                        group_points,
                        initial,
                        seed,
                        max_groups=horizon,
                    )
                elif strategy == "kernel_ivr":
                    t = kernel_selection_order(group_points, initial, seed, horizon)
                else:
                    t = {
                        "strategy": strategy,
                        "seed": seed,
                        "status": "completed",
                        "events": [],
                        "groups": input_selection_order(group_points, initial, strategy, seed)[:horizon],
                    }
                if not set(t["groups"]).issubset(group_points) or t["groups"][: len(initial)] != initial:
                    raise SystemExit("Selection trace violates frozen training pool")
                traces.append({**t, "dataset": name})
    trace_payload = {"protocol_files": freeze["protocol_files"], "traces": traces}
    write_json(OUT / "selection_traces.json", trace_payload)

    # 2) Primary comparison (single scoring).
    result = {
        "protocol_id": "round5-external-frozen-v1",
        "trace_canonical_sha256": canonical(trace_payload),
        "datasets": {},
        "secondary": {},
    }
    for name in DATASETS:
        folder = ROOT / "data/public" / name
        d, s = data[name], splits[name]
        base_run = benchmark(folder / "data.csv", folder / "metadata.json", OUT / name, s)
        train = pools[name]
        calibration = [o for o in d.observations if o.group_id in s["calibration"]]
        corrected = add_residual_correction(build_artifact(replace(d, observations=train), train), train)
        corrected["interval"] = calibrate_interval(corrected, calibration)
        write_json(OUT / name / "model_rbf.json", corrected)
        rbf_predictions = [predict_point(corrected, o.point) for o in tests[name]]
        rbf = metrics(tests[name], rbf_predictions)
        write_csv(
            OUT / name / "rbf_predictions.csv",
            [
                {**asdict(o.point), **p, **{t + "_observed": getattr(o, t) for t in TARGETS}}
                for o, p in zip(tests[name], rbf_predictions)
            ],
        )
        methods = {
            "semiempirical": base_run["metrics"],
            "semiempirical_rbf": rbf,
            "affine": base_run["baseline"].get("metrics"),
            "ahri10": base_run["ahri10_baseline"].get("metrics"),
        }
        status = {"affine": base_run["baseline"]["status"], "ahri10": base_run["ahri10_baseline"]["status"]}
        # Predeclared criteria.
        crit = {"A_p0_numeric": {}, "B_correction": None, "C_vs_strongest_baseline": None}
        for method in ("semiempirical", "semiempirical_rbf"):
            m = methods[method]
            crit["A_p0_numeric"][method] = {
                t: bool(m[t] and m[t]["mare_percent"] <= 5 and m[t]["p95_are_percent"] <= 10) for t in TARGETS
            }
        if any(rbf.get(t) is None or base_run["metrics"].get(t) is None for t in TARGETS):
            crit["B_correction"] = "not_evaluable_missing_predictions"
        else:
            lower = [rbf[t]["mare_percent"] < base_run["metrics"][t]["mare_percent"] for t in TARGETS]
            crit["B_correction"] = (
                "better_on_both" if all(lower) else ("inconsistent" if any(lower) else "worse_on_both")
            )
        reference = "ahri10" if status["ahri10"] == "available" else "affine"
        ref_value, rbf_value = max_mare(methods[reference]), max_mare(rbf)
        verdict = (
            "not_evaluable"
            if ref_value is None or rbf_value is None
            else ("not_worse" if rbf_value <= ref_value else "worse")
        )
        crit["C_vs_strongest_baseline"] = {
            "reference": reference,
            "rbf_max_mare": rbf_value,
            "reference_max_mare": ref_value,
            "verdict": verdict,
        }
        result["datasets"][name] = {
            "counts": {k: len(s[k]) for k in ("train", "calibration", "test")},
            "methods": methods,
            "baseline_status": status,
            "intervals": {"semiempirical": base_run["interval"], "semiempirical_rbf": corrected["interval"]},
            "criteria": crit,
        }
    # 3) Secondary design curves (scored once, after traces frozen above).
    wins, cells = 0, 0
    for name in DATASETS:
        orders = [t for t in traces if t["dataset"] == name]
        rows, failures, summary = evaluate_fixed_orders(
            data[name], pools[name], tests[name], orders, budgets[name], include_residual=True
        )
        write_csv(OUT / name / "design_trials.csv", rows)
        write_csv(OUT / name / "design_summary.csv", summary)
        by = defaultdict(dict)
        for item in summary:
            if item["family"] == "semiempirical_rbf" and item["mass_flow_kg_s_mare_mean_percent"] is not None:
                by[item["budget_groups"]][item["strategy"]] = max(
                    item["mass_flow_kg_s_mare_mean_percent"], item["power_w_mare_mean_percent"]
                )
        comparisons = {}
        for b, values in sorted(by.items()):
            if b in (min(budgets[name]), max(budgets[name])):
                continue  # identical subsets for all strategies at initial and full budgets
            comparisons[b] = values
            if "kernel_ivr" in values and "maximin" in values:
                cells += 1
                wins += values["kernel_ivr"] < values["maximin"]
        result["secondary"][name] = {
            "budgets": budgets[name],
            "records": len(rows),
            "unavailable": len(failures),
            "rbf_max_response_mare_by_budget": comparisons,
            "summary": summary,
        }
    result["secondary"]["kernel_ivr_beats_maximin_cells"] = {
        "wins": wins,
        "cells": cells,
        "verdict": "majority" if wins * 2 > cells else "no_majority",
    }
    write_json(OUT / "result.json", result)
    brief = {
        n: {
            "criteria": v["criteria"],
            **{
                m: (
                    {
                        t: (round(v["methods"][m][t]["mare_percent"], 4) if v["methods"][m].get(t) else None)
                        for t in TARGETS
                    }
                    if v["methods"][m]
                    else None
                )
                for m in v["methods"]
            },
        }
        for n, v in result["datasets"].items()
    }
    print(json.dumps({"primary": brief, "secondary": result["secondary"]["kernel_ivr_beats_maximin_cells"]}, indent=1))


if __name__ == "__main__":
    main()
