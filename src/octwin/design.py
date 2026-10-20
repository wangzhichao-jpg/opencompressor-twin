"""Local D-optimal parameter-information design, not guaranteed error reduction.

Unit independent log-response noise is a declared modelling assumption, not
measurement uncertainty. Repeat groups have equal total influence. This module
scores inputs mathematically; workflow.recommend enforces approval/domain/phase.
"""

import math
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from .core import CompressorSpec, OperatingPoint
from .models import MODEL_VERSION, get_model
from .properties import PROPERTY_VERSION

DEFAULT_RIDGE = 1e-6
DESIGN_VERSION = "local-d-optimal-log-sensitivity-1"


def positive_cost(value):
    if isinstance(value, bool):
        raise ValueError("Relative cost must be finite and positive, not boolean")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Relative cost must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Relative cost must be finite and positive")
    return value


def _context(artifact):
    if artifact.get("prediction_extension") or artifact.get("residual_model") is not None:
        raise ValueError(
            "Parameter D-optimal does not describe the residual extension; use kernel_ivr or the explicit base artifact"
        )
    if artifact.get("model_version") != MODEL_VERSION:
        raise ValueError("D-optimal strategy does not support this model version")
    if artifact.get("environment", {}).get("CoolProp") != PROPERTY_VERSION:
        raise ValueError("D-optimal strategy requires matching property version")
    spec = CompressorSpec(**artifact["compressor"])
    model = get_model(spec.compressor_type)
    if not callable(getattr(model, "log_output_jacobian", None)):
        raise ValueError("Selected model has no supported log-output sensitivity")
    scale = np.asarray(model.bounds[1]) - np.asarray(model.bounds[0])
    return spec, model, scale


def group_information(artifact, points):
    """Mean(J'J) over observations in ONE independent condition group."""
    if not points or any(not isinstance(p, OperatingPoint) for p in points):
        raise ValueError("Information scoring accepts OperatingPoint inputs only, never labels")
    spec, model, scale = _context(artifact)
    matrix = np.zeros((len(scale), len(scale)))
    for point in points:
        jac = np.asarray(model.log_output_jacobian(spec, point, artifact["parameters"])) * scale
        if jac.shape != (2, len(scale)) or not np.isfinite(jac).all():
            raise ValueError("Invalid local log-output sensitivity")
        matrix += jac.T @ jac
    return matrix / len(points)


def training_information(artifact, ridge=DEFAULT_RIDGE):
    if isinstance(ridge, bool) or not isinstance(ridge, (float, int)) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Information ridge must be finite and positive")
    inputs = artifact.get("training_inputs", [])
    groups = artifact.get("training_input_groups")
    if not isinstance(groups, list) or len(groups) != len(inputs) or not inputs:
        raise ValueError(
            "D-optimal requires training_input_groups mapping; recalibrate legacy artifact, do not assume independent repeats"
        )
    if any(not isinstance(g, str) or not g.strip() for g in groups) or set(groups) != set(
        artifact["training_group_ids"]
    ):
        raise ValueError("Invalid training input/group mapping")
    grouped = defaultdict(list)
    for row, group in zip(inputs, groups):
        grouped[group].append(OperatingPoint(**row))
    _, _, scale = _context(artifact)
    matrix = float(ridge) * np.eye(len(scale))
    for points in grouped.values():
        matrix += group_information(artifact, points)
    return matrix


def information_gain(current, addition):
    sign0, log0 = np.linalg.slogdet(current)
    sign1, log1 = np.linalg.slogdet(current + addition)
    if sign0 <= 0 or sign1 <= 0 or not np.isfinite([log0, log1]).all():
        raise ValueError("Invalid positive-definite information matrix")
    difference = float(log1 - log0)
    if difference < -1e-8:
        raise ValueError("Negative information gain beyond numerical tolerance")
    return max(0.0, difference)


def rank_information_groups(artifact, group_points, costs=None, ridge=DEFAULT_RIDGE):
    """Only candidate inputs/current fitted parameters; no candidate output API."""
    if costs is not None and set(costs) != set(group_points):
        raise ValueError("Costs must match every candidate group exactly")
    fisher = training_information(artifact, ridge)
    result = []
    for group, points in group_points.items():
        if not isinstance(group, str) or not group:
            raise ValueError("Candidate group IDs must be nonempty strings")
        cost = positive_cost(costs[group]) if costs is not None else 1.0
        gain = information_gain(fisher, group_information(artifact, points))
        result.append({"group_id": group, "logdet_gain": gain, "relative_cost": cost, "information_score": gain / cost})
    return sorted(result, key=lambda r: r["information_score"], reverse=True)


def recommend_d_optimal(artifact, candidates, budget, costs=None, ridge=DEFAULT_RIDGE):
    """Greedy batch at frozen parameters. Refit only after actual measurement."""
    fisher = training_information(artifact, ridge)
    if costs is not None and set(costs) != {p.point_id for p in candidates}:
        raise ValueError("Costs must match accepted candidate point IDs")
    remaining = [
        (p, group_information(artifact, [p]), positive_cost(costs[p.point_id]) if costs is not None else 1.0)
        for p in candidates
    ]
    selected = []
    for _ in range(min(budget, len(remaining))):
        scores = [(information_gain(fisher, m), cost) for _, m, cost in remaining]
        i = int(np.argmax([gain / cost for gain, cost in scores]))
        point, addition, cost = remaining.pop(i)
        gain, _ = scores[i]
        selected.append(
            {
                **asdict(point),
                "logdet_gain": gain,
                "relative_cost": cost,
                "information_score": gain / cost,
                "reason": "Largest local scaled log-sensitivity logdet gain per declared relative cost",
            }
        )
        fisher += addition
    return {
        "strategy": DESIGN_VERSION,
        "selected": selected,
        "ridge": float(ridge),
        "assumptions": "Unit independent log-response noise; equal condition-group influence; fixed current parameters within a batch",
        "warning": "Local parameter information is not full prediction uncertainty, guaranteed error reduction or bench-time saving; refit after measurement",
    }
