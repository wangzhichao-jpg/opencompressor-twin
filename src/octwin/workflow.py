"""Shared persistence, grouped evaluation, conservative domain and coverage sampling.

Engineering preview: prediction intervals are optional split-conformal relative
residual intervals, not a measurement-uncertainty assessment. No bench control.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

from . import __version__
from .calibration import calibrate
from .core import INPUT_COLUMNS, TARGETS, CompressorSpec, OperatingPoint, load_dataset
from .models import MODEL_VERSION, get_model
from .properties import PROPERTY_VERSION, suction_properties


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    return {
        "octwin": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "CoolProp": PROPERTY_VERSION,
    }


def vector(point):
    return np.array([getattr(point, key) for key in INPUT_COLUMNS[1:]], dtype=float)


def make_split(observations, seed=42):
    groups = sorted({o.group_id for o in observations})
    if len(groups) < 8:
        raise ValueError(
            "Benchmark needs >=8 independent groups; use calibrate for smaller datasets (not blind validation)"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    ntest = max(2, int(len(groups) * 0.2))
    ncal = max(1, int(len(groups) * 0.2))
    return {
        "train": groups[ntest + ncal :],
        "calibration": groups[ntest : ntest + ncal],
        "test": groups[:ntest],
        "seed": seed,
        "method": "seeded_group_random_preview; not a regional holdout",
    }


def check_split(observations, split):
    keys = ("train", "calibration", "test")
    if any(k not in split or not isinstance(split[k], list) for k in keys):
        raise ValueError("Split requires train/calibration/test group lists")
    values = [g for k in keys for g in split[k]]
    if any(not isinstance(g, str) or not g.strip() for g in values):
        raise ValueError("Split groups must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("Train/calibration/test groups overlap or contain duplicates")
    if set(values) != {o.group_id for o in observations}:
        raise ValueError("Split must partition every group exactly once")
    if not split["train"] or not split["test"]:
        raise ValueError("Non-empty train and final test required")


def build_artifact(dataset, observations):
    allowed = {o.point.point_id: o for o in dataset.observations}
    if any(allowed.get(o.point.point_id) != o for o in observations):
        raise ValueError("Training observations must be an unchanged subset of the declared dataset")
    fit = calibrate(dataset.spec, observations)
    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "environment": environment(),
        "compressor": asdict(dataset.spec),
        "dataset_id": dataset.metadata["dataset_id"],
        "dataset_fingerprint": dataset.fingerprint,
        "source_kind": dataset.metadata["source_kind"],
        "validation_role": dataset.metadata["validation_role"],
        "metadata": dataset.metadata,
        **fit,
        "training_group_ids": sorted({o.group_id for o in observations}),
        "training_inputs": [asdict(o.point) for o in observations],
        "training_input_groups": [o.group_id for o in observations],
        "interval": {"status": "unavailable", "reason": "No independent residual calibration groups supplied"},
        "assumptions": [
            "Four-parameter efficiency correlation; not geometry inversion",
            "Efficiency follows metadata power_boundary; no shaft/electrical power conversion",
            "Speed enters throughput only; no separately validated speed-loss terms",
            "Training convex hull is a coverage check, not a safe operating envelope",
            "No real-machine accuracy or saved bench-time claim without matching independent data",
        ],
    }


def load_artifact(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1 or artifact.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported artifact schema/model version")
    if artifact.get("environment", {}).get("CoolProp") != PROPERTY_VERSION:
        raise ValueError("CoolProp version differs from saved artifact; revalidate/refit explicitly")
    spec = CompressorSpec(**artifact["compressor"])
    params = artifact["parameters"]
    model = get_model(spec.compressor_type)
    if set(params) != set(model.parameter_names):
        raise ValueError("Invalid saved parameter names")
    vals = np.array([params[k] for k in model.parameter_names], dtype=float)
    if not np.isfinite(vals).all() or np.any(vals < model.bounds[0]) or np.any(vals > model.bounds[1]):
        raise ValueError("Invalid saved parameters")
    if not artifact.get("training_inputs"):
        raise ValueError("Artifact has no training domain")
    for row in artifact["training_inputs"]:
        OperatingPoint(**row)
    if artifact.get("prediction_extension") or artifact.get("residual_model") is not None:
        from .discrepancy import validate_residual_model

        validate_residual_model(artifact)
    return artifact


def domain(artifact, point):
    train = np.array([vector(OperatingPoint(**p)) for p in artifact["training_inputs"]])
    low, high = train.min(axis=0), train.max(axis=0)
    scale = high - low
    fixed = scale <= np.maximum(np.abs(low), 1) * 1e-10
    scale[fixed] = np.maximum(np.abs(low[fixed]), 1)
    train = (train - low) / scale
    query = (vector(point) - low) / scale
    distance = float(np.linalg.norm(train - query, axis=1).min())
    if np.any(query < -1e-8) or np.any(query[~fixed] > 1 + 1e-8) or np.any(np.abs(query[fixed]) > 1e-8):
        return {"status": "refused_extrapolation", "nearest_distance": distance}
    # Feasible convex combination also detects holes in the affine span of fixed-speed grids.
    feasible = linprog(
        np.zeros(len(train)),
        A_eq=np.vstack([np.ones(len(train)), train.T]),
        b_eq=np.r_[1.0, query],
        bounds=(0, None),
        method="highs",
    )
    if not feasible.success:
        return {"status": "refused_outside_training_hull", "nearest_distance": distance}
    return {
        "status": "covered" if distance < 1e-8 else "in_training_hull",
        "nearest_distance": distance,
        "warning": "Convex hull coverage does not establish model validity or a safe operating envelope",
    }


def predict_point(artifact, point):
    result = {"point_id": point.point_id, **domain(artifact, point)}
    if result["status"].startswith("refused"):
        return result
    try:
        spec = CompressorSpec(**artifact["compressor"])
        result.update(get_model(spec.compressor_type).predict(spec, point, artifact["parameters"]))
        if artifact.get("prediction_extension") or artifact.get("residual_model") is not None:
            from .discrepancy import apply_residual_correction

            result = apply_residual_correction(artifact, point, result)
        if "reference_enthalpy_difference_j_kg" in artifact["metadata"]:
            result["reference_capacity_w"] = result["mass_flow_kg_s"] * float(
                artifact["metadata"]["reference_enthalpy_difference_j_kg"]
            )
            result["reference_capacity_to_power_ratio"] = result["reference_capacity_w"] / result["power_w"]
            if artifact["metadata"]["power_boundary"] == "electrical_input":
                result["reference_cop"] = result["reference_capacity_to_power_ratio"]
        interval = artifact.get("interval", {})
        if interval.get("status") == "available":
            for target in TARGETS:
                radius = interval["relative_radius"][target] * result[target]
                result[target + "_lower"] = max(0.0, result[target] - radius)
                result[target + "_upper"] = result[target] + radius
        else:
            result["interval_status"] = "unavailable"
    except ValueError as exc:
        result.update(status="failed_physics", error=str(exc))
    return result


def calibrate_interval(artifact, observations, coverage=0.9):
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    train_groups = set(artifact["training_group_ids"])
    if train_groups & {o.group_id for o in observations}:
        raise ValueError("Interval calibration overlaps training groups")
    if set(artifact["training_point_ids"]) & {o.point.point_id for o in observations}:
        raise ValueError("Interval calibration overlaps training points")
    scores = {t: {} for t in TARGETS}
    for o in observations:
        prediction = predict_point(artifact, o.point)
        if TARGETS[0] not in prediction:
            return {
                "status": "unavailable",
                "reason": "At least one residual calibration point is outside domain or failed",
                "failed_point_id": o.point.point_id,
            }
        for target in TARGETS:
            score = abs(getattr(o, target) / prediction[target] - 1)
            scores[target][o.group_id] = max(score, scores[target].get(o.group_id, 0))
    n = len(scores[TARGETS[0]])
    k = math.ceil((n + 1) * coverage)
    if n == 0 or k > n:
        return {
            "status": "unavailable",
            "reason": "Too few independent calibration groups for finite conformal quantile",
            "groups": n,
            "nominal_coverage": coverage,
        }
    return {
        "status": "available",
        "method": "group-max split conformal, relative-to-prediction residual",
        "groups": n,
        "nominal_coverage": coverage,
        "relative_radius": {t: float(sorted(scores[t].values())[k - 1]) for t in TARGETS},
        "calibration_group_ids": sorted(scores[TARGETS[0]]),
        "limitation": "Coverage assumes exchangeable independent groups; not guaranteed on regional/shifted data; not measurement uncertainty",
    }


def metrics(observations, predictions):
    if len(observations) != len(predictions):
        raise ValueError("One prediction/status record per evaluated observation is required")
    valid = [(o, p) for o, p in zip(observations, predictions) if all(t in p for t in TARGETS)]
    output = {
        "total": len(observations),
        "predicted": len(valid),
        "refused_or_failed": len(observations) - len(valid),
        "note": "Error/interval statistics conditional on predicted points; failures retained in denominator above",
    }
    for target in TARGETS:
        if not valid:
            output[target] = None
            continue
        err = np.array([abs(p[target] / getattr(o, target) - 1) * 100 for o, p in valid])
        item = {
            "mare_percent": float(err.mean()),
            "p95_are_percent": float(np.percentile(err, 95)),
            "max_are_percent": float(err.max()),
        }
        pairs = [(o, p) for o, p in valid if target + "_lower" in p]
        if pairs:
            item["interval_coverage"] = float(
                np.mean([p[target + "_lower"] <= getattr(o, target) <= p[target + "_upper"] for o, p in pairs])
            )
            item["interval_mean_relative_width"] = float(
                np.mean([(p[target + "_upper"] - p[target + "_lower"]) / p[target] for o, p in pairs])
            )
            item["interval_points"] = len(pairs)
        output[target] = item
    return output


def linear_baseline(train, test, predictions):
    """Affine input response surface, rank-reduced using training-only SVD."""
    x = np.array([vector(o.point) for o in train])
    low, scale = x.mean(axis=0), np.ptp(x, axis=0)
    fixed = scale == 0
    # Mean summation can leave a tiny constant residual at non-integer setpoints.
    # Remove exactly fixed columns before SVD; otherwise rank is overstated.
    low[fixed] = x[0, fixed]
    scale[fixed] = 1
    z = (x - low) / scale
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    rank = int(np.linalg.matrix_rank(z))
    basis = vt[:rank].T
    design = np.column_stack([np.ones(len(z)), z @ basis])
    if len({o.group_id for o in train}) <= rank + 1:
        return {"status": "unavailable", "reason": "Insufficient independent training groups for affine baseline"}
    # Same equal-repeat-group weighting as the semiempirical estimator.
    weights = np.array([sum(a.group_id == o.group_id for a in train) ** -0.5 for o in train])
    y = np.array([[getattr(o, t) for t in TARGETS] for o in train])
    coef = np.linalg.lstsq(design * weights[:, None], y * weights[:, None], rcond=None)[0]
    pred = []
    for o, matched in zip(test, predictions):
        row = {"point_id": o.point.point_id, "status": matched["status"]}
        if matched["status"] in {"covered", "in_training_hull"}:
            val = np.r_[1.0, ((vector(o.point) - low) / scale) @ basis] @ coef
            if np.isfinite(val).all() and (val > 0).all():
                row.update(dict(zip(TARGETS, map(float, val))))
            else:
                row["status"] = "failed_nonpositive_baseline"
        pred.append(row)
    return {
        "status": "available",
        "family": "training-only affine input surface",
        "rank": rank + 1,
        "metrics": metrics(test, pred),
        "predictions": pred,
    }


def stratified_metrics(train, test, predictions):
    """Predeclared three-bin pressure slices; inputs set edges, never errors."""
    output = {}
    for field in ("suction_pressure_pa", "discharge_pressure_pa"):
        low = min(getattr(o.point, field) for o in train)
        high = max(getattr(o.point, field) for o in train)
        if high == low:
            continue
        edges = np.linspace(low, high, 4)
        slices = []
        for i in range(3):
            indices = [
                j
                for j, o in enumerate(test)
                if edges[i] <= getattr(o.point, field)
                and (getattr(o.point, field) <= edges[i + 1] if i == 2 else getattr(o.point, field) < edges[i + 1])
            ]
            if indices:
                slices.append(
                    {
                        "low_pa": float(edges[i]),
                        "high_pa": float(edges[i + 1]),
                        "point_ids": [test[j].point.point_id for j in indices],
                        "metrics": metrics([test[j] for j in indices], [predictions[j] for j in indices]),
                    }
                )
        outside = [j for j, o in enumerate(test) if not low <= getattr(o.point, field) <= high]
        output[field] = {
            "binning": "three equal-width bins from training inputs only",
            "slices": slices,
            "outside_training_range": len(outside),
        }
    return output


def benchmark(csv_path, meta_path, output, split=None, seed=42):
    dataset = load_dataset(csv_path, meta_path)
    split = make_split(dataset.observations, seed) if split is None else split
    check_split(dataset.observations, split)
    existing_result = Path(output) / "result.json"
    if existing_result.exists():
        previous = json.loads(existing_result.read_text(encoding="utf-8"))
        if previous.get("dataset_fingerprint") != dataset.fingerprint or previous.get("split") != split:
            raise ValueError(
                "Completed run has a different dataset/split; use a new output directory, do not replace frozen evidence"
            )
    subsets = {k: [o for o in dataset.observations if o.group_id in split[k]] for k in ("train", "calibration", "test")}
    artifact = build_artifact(dataset, subsets["train"])
    artifact["interval"] = calibrate_interval(artifact, subsets["calibration"])
    predictions = [predict_point(artifact, o.point) for o in subsets["test"]]
    evidence = {
        "synthetic": "SOFTWARE SELF-TEST ONLY - not real-machine validation",
        "manufacturer_map": "MANUFACTURER MAP REPRODUCTION - not independent experimental validation",
        "digitized_figure": "DIGITIZED FIGURE COMPARISON - source nature and extraction uncertainty require review",
        "experiment": "EXPERIMENTAL DATA COMPARISON - see source review and declared scope limitations",
    }[dataset.metadata["source_kind"]]
    from .baselines import evaluate_ahri10

    ahri10 = evaluate_ahri10(dataset.spec, subsets["train"], subsets["test"], predictions)
    source_limits = dataset.metadata.get("known_limitations", [])
    if isinstance(source_limits, str):
        source_limits = [source_limits]
    result = {
        "dataset_id": dataset.metadata["dataset_id"],
        "compressor": asdict(dataset.spec),
        "evidence_label": evidence,
        "source_kind": dataset.metadata["source_kind"],
        "validation_role": dataset.metadata["validation_role"],
        "dataset_fingerprint": dataset.fingerprint,
        "source": dataset.metadata["source"],
        "license": dataset.metadata["license"],
        "power_boundary": dataset.metadata["power_boundary"],
        "environment": environment(),
        "split": split,
        "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        "group_counts": {k: len(split[k]) for k in subsets},
        "metrics": metrics(subsets["test"], predictions),
        "training_fit_only_NOT_validation": metrics(
            subsets["train"], [predict_point(artifact, o.point) for o in subsets["train"]]
        ),
        "stratified_metrics": stratified_metrics(subsets["train"], subsets["test"], predictions),
        "source_review": {
            "speed_definition": dataset.metadata["speed_definition"],
            "nominal_speed_provenance": dataset.metadata.get("nominal_speed_provenance"),
            "admission_status": dataset.metadata.get("admission_status", "not_formally_admitted"),
            "source_location": dataset.metadata.get("source_location", "see manifest"),
            "known_limitations": source_limits,
        },
        "baseline": linear_baseline(subsets["train"], subsets["test"], predictions),
        "ahri10_baseline": ahri10,
        "interval": artifact["interval"],
        "diagnostics": artifact["diagnostics"],
        "test_predictions": [
            {**asdict(o.point), **p, "group_id": o.group_id, **{t + "_observed": getattr(o, t) for t in TARGETS}}
            for o, p in zip(subsets["test"], predictions)
        ],
        "limitations": artifact["assumptions"]
        + [
            "Single grouped split; not repeated active-learning evaluation",
            "AHRI-style baseline status: " + ahri10["status"] + "; no measured-time benefit evaluated",
            "Final test observations never used for calibration or recommendation",
        ]
        + source_limits,
    }
    output = Path(output)
    write_json(output / "model.json", artifact)
    write_json(output / "split.json", split)
    write_json(output / "result.json", result)
    from .reporting import report

    report(result, output / "report.html")
    from .reporting import write_csv

    write_csv(output / "predictions.csv", result["test_predictions"])
    if ahri10["status"] == "available":
        write_json(
            output / "ahri10_model.json",
            {
                "compressor": asdict(dataset.spec),
                "environment": environment(),
                **{k: v for k, v in ahri10.items() if k not in {"predictions", "metrics"}},
            },
        )
        write_csv(output / "ahri10_predictions.csv", ahri10["predictions"])
    else:
        write_json(output / "ahri10_model.json", ahri10)
        write_csv(output / "ahri10_predictions.csv", [{"status": "unavailable", "reason": ahri10.get("reason", "")}])
    return result


def recommend(artifact, candidates, budget=1, strategy="coverage", costs=None):
    """Approved inputs only; parameter information or full-input prior variance."""
    if strategy not in {"coverage", "d_optimal", "kernel_ivr"}:
        raise ValueError("Recommendation strategy must be coverage, d_optimal or kernel_ivr")
    if any(not isinstance(p, OperatingPoint) for p in candidates):
        raise ValueError("Recommendation accepts OperatingPoint inputs only, never observed labels")
    if costs is not None:
        from .design import positive_cost

        if strategy not in {"d_optimal", "kernel_ivr"}:
            raise ValueError("Relative costs apply only to d_optimal/kernel_ivr strategy")
        if not isinstance(costs, dict) or set(costs) != {p.point_id for p in candidates}:
            raise ValueError("Costs must match all supplied candidate IDs exactly")
        costs = {k: positive_cost(v) for k, v in costs.items()}
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if len({p.point_id for p in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate point_id")
    accepted, rejected, seen_inputs = [], [], set()
    spec = CompressorSpec(**artifact["compressor"])
    training_ids = set(artifact["training_point_ids"])
    # IDs of candidates are informational only; exact already-observed inputs also excluded.
    for p in candidates:
        d = domain(artifact, p)
        if p.point_id in training_ids or d["nearest_distance"] < 1e-8:
            rejected.append({"point_id": p.point_id, "reason": "already observed training input"})
            continue
        if d["status"].startswith("refused"):
            rejected.append({"point_id": p.point_id, "reason": d["status"]})
            continue
        try:
            suction_properties(p, spec.refrigerant)
        except ValueError as exc:
            rejected.append({"point_id": p.point_id, "reason": str(exc)})
            continue
        key = tuple(vector(p))
        if key in seen_inputs:
            rejected.append(
                {"point_id": p.point_id, "reason": "duplicate candidate input, not another independent condition"}
            )
            continue
        seen_inputs.add(key)
        accepted.append(p)
    if strategy in {"d_optimal", "kernel_ivr"}:
        if strategy == "d_optimal":
            from .design import recommend_d_optimal as recommender
        else:
            from .kernels import recommend_kernel_ivr as recommender
        accepted_costs = {p.point_id: costs[p.point_id] for p in accepted} if costs is not None else None
        result = recommender(artifact, accepted, budget, accepted_costs)
        result["rejected"] = rejected
        result["candidate_approval"] = (
            "External engineer-approved input set required; model coverage is not safety approval"
        )
        return result
    reference = np.array([vector(OperatingPoint(**p)) for p in artifact["training_inputs"]])
    scale = np.ptp(reference, axis=0)
    scale[scale == 0] = 1
    selected = []
    for _ in range(min(budget, len(accepted))):
        scores = [float(np.linalg.norm((reference - vector(p)) / scale, axis=1).min()) for p in accepted]
        index = int(np.argmax(scores))
        p = accepted.pop(index)
        selected.append(
            {
                **asdict(p),
                "coverage_distance": scores[index],
                "reason": "Largest normalized distance from currently observed/selected inputs",
            }
        )
        reference = np.vstack([reference, vector(p)])
    return {
        "strategy": "greedy_maximin_coverage_preview",
        "selected": selected,
        "rejected": rejected,
        "warning": "Approved candidate set required; coverage is not information gain or safety approval; no output labels/cost model used",
    }
