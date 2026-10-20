"""Revealed-label adaptive replay. No calibration/final labels in selector API."""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import Dataset, load_dataset
from .design import DEFAULT_RIDGE, DESIGN_VERSION, rank_information_groups
from .evaluation import evaluate_fixed_orders, input_selection_order
from .models import MODEL_VERSION
from .properties import suction_properties
from .reporting import curve_report, write_csv
from .workflow import build_artifact, check_split, domain, environment, write_json


class TrainingPoolOracle:
    """Simulation boundary, not a cryptographic vault. Holds TRAINING POOL only."""

    def __init__(self, observations):
        self._groups = defaultdict(list)
        for observed in observations:
            self._groups[observed.group_id].append(observed)
        self.revealed_groups = []
        self._observed = []

    def reveal(self, group):
        if group not in self._groups:
            raise ValueError("Requested group is not in the approved training pool")
        if group in self.revealed_groups:
            raise ValueError("Group already revealed; no double-counted acquisition")
        self.revealed_groups.append(group)
        self._observed.extend(self._groups[group])

    def observations(self):
        return list(self._observed)


def adaptive_information_order(
    spec, metadata, fingerprint, oracle, group_points, initial, seed, ridge=DEFAULT_RIDGE, max_groups=None
):
    """Revealed-only fits; stop acquisition at the declared horizon, not pool end."""
    limit = len(group_points) if max_groups is None else max_groups
    if isinstance(limit, bool) or not isinstance(limit, int) or not len(initial) <= limit <= len(group_points):
        raise ValueError("Acquisition horizon must be between initial and training-pool group counts")
    for group in initial:
        oracle.reveal(group)
    order = list(initial)
    remaining = sorted(set(group_points) - set(initial))
    np.random.default_rng(seed).shuffle(remaining)
    events = []
    fit_attempts = 0
    while remaining and len(order) < limit:
        observed = oracle.observations()
        try:
            view = Dataset(spec, observed, metadata, fingerprint)
            fit_attempts += 1
            artifact = build_artifact(view, observed)
            ranked = rank_information_groups(artifact, {g: group_points[g] for g in remaining}, ridge=ridge)
            chosen = ranked[0]
        except ValueError as exc:
            return {
                "strategy": "d_optimal",
                "seed": seed,
                "groups": order,
                "status": "stopped_selector_failed",
                "error": str(exc),
                "events": events,
                "fit_attempts": fit_attempts,
                "reveal_log": list(oracle.revealed_groups),
            }
        group = chosen["group_id"]
        events.append(
            {
                "observed_groups_before_selection": list(order),
                "selected_group_id": group,
                "candidate_count": len(remaining),
                "parameters_before_selection": artifact["parameters"],
                "local_fit_rank": artifact["diagnostics"]["rank"],
                "logdet_gain": chosen["logdet_gain"],
                "relative_cost": chosen["relative_cost"],
                "information_score": chosen["information_score"],
                "point_ids": [p.point_id for p in group_points[group]],
            }
        )
        # Reveal only AFTER selecting. No output is passed to rank_information_groups.
        oracle.reveal(group)
        order.append(group)
        remaining.remove(group)
    return {
        "strategy": "d_optimal",
        "seed": seed,
        "groups": order,
        "status": "completed",
        "events": events,
        "fit_attempts": fit_attempts,
        "reveal_log": list(oracle.revealed_groups),
    }


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_adaptive_replay(csv_path, meta_path, split, output, budgets=(9, 12, 16, 24, 32, 40), seeds=tuple(range(10))):
    data = load_dataset(csv_path, meta_path)
    check_split(data.observations, split)
    initial = split.get("mandatory_hull_training_groups")
    if (
        not isinstance(initial, list)
        or len(initial) < 4
        or len(initial) != len(set(initial))
        or not set(initial).issubset(split["train"])
    ):
        raise ValueError("Adaptive replay requires >=4 unique initial hull groups inside frozen training pool")
    pool = [o for o in data.observations if o.group_id in split["train"]]
    test = [o for o in data.observations if o.group_id in split["test"]]
    group_points = defaultdict(list)
    for observed in pool:
        group_points[observed.group_id].append(observed.point)
        suction_properties(observed.point, data.spec.refrigerant)
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(
            isinstance(b, bool) or not isinstance(b, int) or not len(initial) <= b <= len(group_points) for b in budgets
        )
    ):
        raise ValueError("Adaptive budgets must be unique integers within initial/training-pool group counts")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("Adaptive seeds must be unique nonnegative integers")
    initial_domain = {"training_inputs": [asdict(p) for g in initial for p in group_points[g]]}
    outside = [
        p.point_id
        for points in group_points.values()
        for p in points
        if domain(initial_domain, p)["status"].startswith("refused")
    ]
    if outside:
        raise ValueError(
            "Initial groups do not cover the candidate training pool; freeze a valid hull protocol before replay"
        )
    settings = {
        "dataset_fingerprint": data.fingerprint,
        "split": split,
        "budgets": list(budgets),
        "seeds": list(seeds),
        "strategies": ["random", "maximin", "d_optimal"],
        "design_version": DESIGN_VERSION,
        "model_version": MODEL_VERSION,
        "ridge": DEFAULT_RIDGE,
        "cost": "one relative cost per independent condition group; no timing model",
        "parameter_scaling": "declared parameter bounds range",
        "noise_assumption": "unit independent log-response noise",
        "initial_groups": initial,
        "evaluation_status": "secondary retrospective adaptive replay, NOT new blind validation",
        "selector_boundary": "training-pool oracle only; candidates are input-only; final/calibration outputs never provided to selector",
        "evaluation_order": "freeze settings, generate/freeze complete selection traces, THEN score fixed final test at budgets",
    }
    output = Path(output)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing adaptive protocol differs; use a new directory, do not overwrite frozen evidence")
    write_json(protocol_path, settings)
    traces = []
    for strategy in ("random", "maximin", "d_optimal"):
        for seed in seeds:
            if strategy == "d_optimal":
                oracle = TrainingPoolOracle(pool)  # Deliberately excludes calibration/final observations.
                trace = adaptive_information_order(
                    data.spec,
                    data.metadata,
                    data.fingerprint,
                    oracle,
                    group_points,
                    initial,
                    seed,
                    max_groups=max(budgets),
                )
            else:
                trace = {
                    "strategy": strategy,
                    "seed": seed,
                    "groups": input_selection_order(group_points, initial, strategy, seed)[: max(budgets)],
                    "status": "completed",
                    "events": [],
                    "reveal_log": "inputs-only fixed order; outputs revealed only for budgeted fits",
                }
            selected = trace["groups"]
            if len(selected) != len(set(selected)) or not set(selected).issubset(group_points):
                raise ValueError("Selection trace leaves approved training pool or repeats a group")
            if selected[: len(initial)] != initial or (
                trace["status"] == "completed" and len(selected) != max(budgets)
            ):
                raise ValueError("Selection trace changed initial groups or omitted a completed acquisition")
            if strategy == "d_optimal" and trace["reveal_log"] != selected:
                raise ValueError("Adaptive reveal ledger differs from selected groups")
            traces.append(trace)
    trace_payload = {"protocol_sha256": _canonical_hash(settings), "traces": traces}
    trace_path = output / "selection_traces.json"
    if trace_path.exists() and json.loads(trace_path.read_text(encoding="utf-8")) != trace_payload:
        raise ValueError("Existing frozen selection trace differs; preserve old trace and use a new output directory")
    write_json(trace_path, trace_payload)
    # All strategies have finished selecting before final answers are scored.
    for observed in test:
        suction_properties(observed.point, data.spec.refrigerant)
    rows, failures, summary = evaluate_fixed_orders(data, pool, test, traces, budgets)
    source_limits = data.metadata.get("known_limitations", [])
    if isinstance(source_limits, str):
        source_limits = [source_limits]
    selected_groups = {g for trace in traces for g in trace["groups"]}
    calibration_seen = len(selected_groups.intersection(split["calibration"]))
    final_seen = len(selected_groups.intersection(split["test"]))
    audit = {
        "strategies": settings["strategies"],
        "seeds": len(seeds),
        "initial_groups": len(initial),
        "max_revealed_groups": max(budgets),
        "total_orders": len(traces),
        "selector_stops": sum(t["status"] != "completed" for t in traces),
        "adaptive_refits": sum(len(t["events"]) for t in traces),
        "adaptive_fit_attempts": sum(t.get("fit_attempts", 0) for t in traces),
        "unique_selection_orders": {
            s: len({_canonical_hash(t["groups"]) for t in traces if t["strategy"] == s}) for s in settings["strategies"]
        },
        "calibration_groups_used_by_selector": calibration_seen,
        "final_groups_used_by_selector": final_seen,
        "trace_canonical_json_sha256": _canonical_hash(trace_payload),
        "trace_file_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    result = {
        "kind": "adaptive_replay",
        "dataset_id": data.metadata["dataset_id"],
        "source_kind": data.metadata["source_kind"],
        "source": data.metadata["source"],
        "validation_role": data.metadata["validation_role"],
        "compressor": asdict(data.spec),
        "power_boundary": data.metadata["power_boundary"],
        "speed_definition": data.metadata["speed_definition"],
        "environment": environment(),
        "dataset_fingerprint": data.fingerprint,
        "protocol_sha256": _canonical_hash(settings),
        "selection_audit": audit,
        "rows": rows,
        "failures": failures,
        "summary": summary,
        "limitations": [
            "Local D-optimal targets four-parameter information, NOT model discrepancy or guaranteed prediction-error reduction",
            "Current correlation sensitivity depends on pressure ratio only; absolute-pressure/thermal discrepancy is invisible to this selector",
            "D-optimal is tailored to the semiempirical model; baselines share the acquired subset, not their own optimal-design policies",
            "Unit log-response noise and fixed ridge are design assumptions, not measurement uncertainty",
            "All adaptive orders generated before final scoring; final set is reused secondary analysis, not new prospective blind validation",
            "Initial hull groups count toward budget; calibration/final group costs excluded; point-only curves, no total-time savings",
            "Repeated seeds may produce identical orders; report unique orders, not independent physical replicates",
            "No hidden fallback when fitting/selection fails; later budgets retain unavailable status",
        ]
        + source_limits,
    }
    write_json(output / "result.json", result)
    write_csv(output / "trials.csv", rows)
    write_csv(output / "summary.csv", summary)
    curve_report(result, output / "report.html")
    return result
