"""AHRI-540-style ten-coefficient baseline; never import source fitted maps.

Full bivariate cubic in dew temperatures: 1,Te,Td,Te^2,Te*Td,Td^2,
Te^3,Te^2*Td,Te*Td^2,Td^3. Training-only affine scaling of temperatures
preserves that polynomial space. This is a declared comparison expression,
not an assertion of certification/compliance with any edition of AHRI540.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from CoolProp.CoolProp import PropsSI

from .core import TARGETS
from .properties import PROPERTY_VERSION


@lru_cache(maxsize=8192)
def _dew_pair(ps, pd, fluid):
    try:
        pair = np.array([PropsSI("T", "P", ps, "Q", 1, fluid), PropsSI("T", "P", pd, "Q", 1, fluid)])
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"AHRI-style dew temperatures unavailable: {exc}") from exc
    if not np.isfinite(pair).all():
        raise ValueError("Invalid dew temperatures")
    return tuple(map(float, pair))


def dew_features(spec, point):
    return np.asarray(_dew_pair(point.suction_pressure_pa, point.discharge_pressure_pa, spec.refrigerant))


def cubic_design(features):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != 2 or not np.isfinite(features).all():
        raise ValueError("Expected finite (n,2) temperature features")
    e, d = features.T
    return np.column_stack([np.ones(len(e)), e, d, e * e, e * d, d * d, e**3, e * e * d, e * d * d, d**3])


def fit_ahri10(spec, observations):
    """Independent per-response relative LS; repeats have equal group influence."""
    if len({o.group_id for o in observations}) <= 10:
        raise ValueError("AHRI10 requires >10 independent training groups, not repeated row count")
    temperatures = np.array([o.point.suction_temperature_k for o in observations])
    speeds = np.array([o.point.speed_rpm for o in observations])
    if np.ptp(temperatures) > 1e-6 or np.ptp(speeds) > 1e-6:
        raise ValueError("AHRI10 preview requires fixed suction temperature and fixed speed; no mixed regimes/speeds")
    features = np.array([dew_features(spec, o.point) for o in observations])
    center = features.mean(axis=0)
    scale = np.ptp(features, axis=0)
    if np.any(scale <= 1e-10):
        raise ValueError("AHRI10 requires variation in both dew temperatures")
    design = cubic_design((features - center) / scale)
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design, tol=singular[0] * 1e-10))
    if rank != 10:
        raise ValueError(f"AHRI10 design rank {rank}/10; full cubic is not identifiable")
    condition = float(singular[0] / singular[-1])
    if condition > 1e8:
        raise ValueError("AHRI10 design too ill-conditioned")
    counts = {g: sum(o.group_id == g for o in observations) for g in {o.group_id for o in observations}}
    group_weights = np.array([counts[o.group_id] ** -0.5 for o in observations])
    coefficients = {}
    for target in TARGETS:
        y = np.array([getattr(o, target) for o in observations])
        weight = group_weights / y
        coefficients[target] = np.linalg.lstsq(design * weight[:, None], group_weights, rcond=None)[0].tolist()
    return {
        "family": "AHRI540-style full bivariate cubic (10 coefficients per response)",
        "version": 1,
        "property_version": PROPERTY_VERSION,
        "center_k": center.tolist(),
        "scale_k": scale.tolist(),
        "coefficients": coefficients,
        "fixed_suction_temperature_k": float(temperatures[0]),
        "fixed_speed_rpm": float(speeds[0]),
        "diagnostics": {"rank": rank, "condition_number": condition, "training_groups": len(counts)},
        "weighting": "relative residual least squares; equal group weight, no measurement-uncertainty interpretation",
        "training_point_ids": [o.point.point_id for o in observations],
        "limitation": "Point baseline only, intervals not calibrated; original published map coefficients never used",
    }


def predict_ahri10(spec, point, fitted):
    if fitted.get("version") != 1 or fitted.get("property_version") != PROPERTY_VERSION:
        raise ValueError("Unsupported AHRI10/property version; refit or reproduce the frozen environment")
    if (
        abs(point.suction_temperature_k - fitted["fixed_suction_temperature_k"]) > 1e-6
        or abs(point.speed_rpm - fitted["fixed_speed_rpm"]) > 1e-6
    ):
        raise ValueError("AHRI10 prediction regime differs from training")
    features = (dew_features(spec, point) - np.array(fitted["center_k"])) / np.array(fitted["scale_k"])
    row = cubic_design(features[None, :])[0]
    values = {t: float(row @ np.array(fitted["coefficients"][t])) for t in TARGETS}
    if not all(np.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError("AHRI10 produced nonpositive/nonfinite response")
    return values


def evaluate_ahri10(spec, train, test, matched_domain):
    from .workflow import metrics

    try:
        fitted = fit_ahri10(spec, train)
    except ValueError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    predictions = []
    for observed, matched in zip(test, matched_domain):
        result = {"point_id": observed.point.point_id, "status": matched["status"]}
        if matched["status"] in {"covered", "in_training_hull"}:
            try:
                result.update(predict_ahri10(spec, observed.point, fitted))
            except ValueError as exc:
                result.update(status="failed_ahri10", error=str(exc))
        predictions.append(result)
    return {"status": "available", **fitted, "metrics": metrics(test, predictions), "predictions": predictions}
