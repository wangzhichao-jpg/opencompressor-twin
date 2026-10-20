"""Full-input RBF discrepancy-prior design; NOT calibrated physical uncertainty.

Mathematical helpers accept inputs only. workflow.recommend provides the common
phase/domain/approval checks. Group centers are design representatives, not new
measured observations. Fixed hyperparameters are engineering research choices.
"""

import math
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from .core import INPUT_COLUMNS, OperatingPoint
from .design import positive_cost

LENGTH_SCALE = 0.5
KERNEL_RIDGE = 0.01
KERNEL_VERSION = "rbf-input-integrated-variance-1"


def values(point):
    return np.array([getattr(point, k) for k in INPUT_COLUMNS[1:]], dtype=float)


def normalizer(points):
    rows = np.array([values(p) for p in points])
    if not len(rows):
        raise ValueError("Kernel requires nonempty training inputs")
    low, high = rows.min(axis=0), rows.max(axis=0)
    scale = high - low
    fixed = scale <= np.maximum(np.abs(low), 1) * 1e-10
    scale[fixed] = np.maximum(np.abs(low[fixed]), 1)
    return low, scale


def rbf(x, y, length_scale=LENGTH_SCALE):
    if (
        isinstance(length_scale, bool)
        or not isinstance(length_scale, (int, float))
        or not math.isfinite(length_scale)
        or length_scale <= 0
    ):
        raise ValueError("Kernel length scale must be finite and positive")
    return np.exp(-0.5 * np.sum((np.asarray(x)[:, None, :] - np.asarray(y)[None, :, :]) ** 2, axis=2) / length_scale**2)


def training_groups(artifact):
    rows, ids = artifact.get("training_inputs", []), artifact.get("training_input_groups")
    if (
        not rows
        or not isinstance(ids, list)
        or len(rows) != len(ids)
        or any(not isinstance(g, str) or not g.strip() for g in ids)
        or set(ids) != set(artifact["training_group_ids"])
    ):
        raise ValueError("Kernel requires valid training_input_groups; recalibrate legacy artifact")
    groups = defaultdict(list)
    for row, g in zip(rows, ids):
        groups[g].append(OperatingPoint(**row))
    return groups


def group_centers(groups, low, scale):
    if any(
        not isinstance(g, str) or not g or not ps or any(not isinstance(p, OperatingPoint) for p in ps)
        for g, ps in groups.items()
    ):
        raise ValueError("Kernel groups must contain OperatingPoint inputs only, never labels")
    return np.array([(np.mean([values(p) for p in ps], axis=0) - low) / scale for ps in groups.values()])


def posterior_covariance(target, observed, ridge=KERNEL_RIDGE, length_scale=LENGTH_SCALE):
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Kernel ridge must be finite and positive")
    prior = rbf(target, target, length_scale)
    cross = rbf(target, observed, length_scale)
    result = prior - cross @ np.linalg.solve(
        rbf(observed, observed, length_scale) + ridge * np.eye(len(observed)), cross.T
    )
    result = (result + result.T) / 2
    if not np.isfinite(result).all() or np.min(np.diag(result)) < -1e-8:
        raise ValueError("Invalid kernel posterior covariance")
    return result


def _design_state(artifact, groups, ridge, length_scale):
    observed = training_groups(artifact)
    if set(observed).intersection(groups):
        raise ValueError("Kernel candidate groups overlap already acquired groups")
    low, scale = normalizer([p for ps in observed.values() for p in ps])
    train = group_centers(observed, low, scale)
    all_groups = dict(observed)
    all_groups.update(groups)
    target = group_centers(all_groups, low, scale)
    covariance = posterior_covariance(target, train, ridge, length_scale)
    indices = {g: i for i, g in enumerate(all_groups)}
    return covariance, indices


def rank_kernel_groups(artifact, groups, costs=None, ridge=KERNEL_RIDGE, length_scale=LENGTH_SCALE):
    """Integrated latent variance reduction over observed+candidate INPUT groups."""
    if costs is not None and set(costs) != set(groups):
        raise ValueError("Costs must match kernel candidate groups")
    covariance, indices = _design_state(artifact, groups, ridge, length_scale)
    result = []
    for g in groups:
        i = indices[g]
        cost = positive_cost(costs[g]) if costs is not None else 1.0
        gain = float(np.mean(covariance[:, i] ** 2) / (covariance[i, i] + ridge))
        result.append(
            {"group_id": g, "integrated_variance_gain": gain, "relative_cost": cost, "information_score": gain / cost}
        )
    return sorted(result, key=lambda r: r["information_score"], reverse=True)


def kernel_selection_order(group_points, initial, seed, max_groups):
    """Fixed-hyperparameter INPUT-ONLY design, not output-adaptive discovery."""
    if not initial or len(set(initial)) != len(initial) or not set(initial).issubset(group_points):
        raise ValueError("Kernel initial groups must be unique training-pool inputs")
    if (
        isinstance(max_groups, bool)
        or not isinstance(max_groups, int)
        or not len(initial) <= max_groups <= len(group_points)
    ):
        raise ValueError("Invalid kernel acquisition horizon")
    artifact = {
        "training_inputs": [asdict(p) for g in initial for p in group_points[g]],
        "training_input_groups": [g for g in initial for p in group_points[g]],
        "training_group_ids": initial,
    }
    remaining = sorted(set(group_points) - set(initial))
    np.random.default_rng(seed).shuffle(remaining)
    covariance, indices = _design_state(artifact, {g: group_points[g] for g in remaining}, KERNEL_RIDGE, LENGTH_SCALE)
    order, events = list(initial), []
    while remaining and len(order) < max_groups:
        scores = [
            float(np.mean(covariance[:, indices[g]] ** 2) / (covariance[indices[g], indices[g]] + KERNEL_RIDGE))
            for g in remaining
        ]
        index = int(np.argmax(scores))
        group = remaining.pop(index)
        events.append(
            {
                "selected_group_id": group,
                "observed_groups_before_selection": list(order),
                "integrated_variance_gain": scores[index],
                "relative_cost": 1.0,
                "information_score": scores[index],
            }
        )
        i = indices[group]
        column = covariance[:, i].copy()
        covariance -= np.outer(column, column) / (covariance[i, i] + KERNEL_RIDGE)
        order.append(group)
    return {
        "strategy": "kernel_ivr",
        "seed": seed,
        "groups": order,
        "status": "completed",
        "events": events,
        "reveal_log": "input-only fixed-kernel order; outputs revealed only by budgeted evaluators",
    }


def recommend_kernel_ivr(artifact, candidates, budget, costs=None, ridge=KERNEL_RIDGE, length_scale=LENGTH_SCALE):
    groups = {p.point_id: [p] for p in candidates}
    if costs is not None and set(costs) != set(groups):
        raise ValueError("Costs must match accepted kernel candidate IDs")
    # Candidate point IDs need not be nominal group IDs; use disjoint design IDs.
    mapping = {"candidate:" + k: v for k, v in groups.items()}
    covariance, indices = _design_state(artifact, mapping, ridge, length_scale)
    remaining = list(groups)
    selected = []
    for _ in range(min(budget, len(remaining))):
        scores = []
        for g in remaining:
            i = indices["candidate:" + g]
            gain = float(np.mean(covariance[:, i] ** 2) / (covariance[i, i] + ridge))
            cost = positive_cost(costs[g]) if costs is not None else 1.0
            scores.append((gain / cost, gain, cost))
        chosen = int(np.argmax([s[0] for s in scores]))
        g = remaining.pop(chosen)
        score, gain, cost = scores[chosen]
        i = indices["candidate:" + g]
        column = covariance[:, i].copy()
        covariance -= np.outer(column, column) / (covariance[i, i] + ridge)
        selected.append(
            {
                **asdict(groups[g][0]),
                "integrated_variance_gain": gain,
                "relative_cost": cost,
                "information_score": score,
                "reason": "Largest input-kernel integrated latent variance reduction per relative cost",
            }
        )
    return {
        "strategy": KERNEL_VERSION,
        "selected": selected,
        "ridge": ridge,
        "length_scale": length_scale,
        "assumptions": "Fixed unit-amplitude RBF discrepancy prior; one effective support per condition group; integration over approved observed+candidate inputs",
        "warning": "Prior-dependent design score, NOT calibrated prediction intervals, guaranteed error reduction or bench-time saving",
    }
