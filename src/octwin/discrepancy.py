"""Explicit RBF log-residual extension. Research model, not geometry or GUM.

Only revealed training observations enter the residual fit. Parameters are fixed
before development CV. Original four-parameter equations remain unchanged.
"""

import math
from collections import defaultdict
from copy import deepcopy

import numpy as np

from .core import TARGETS, CompressorSpec
from .kernels import KERNEL_RIDGE, LENGTH_SCALE, group_centers, normalizer, rbf
from .models import get_model

EXTENSION_VERSION = "rbf-log-residual-1"


def add_residual_correction(artifact, observations):
    if artifact.get("prediction_extension") or artifact.get("residual_model"):
        raise ValueError("Residual correction already present; fit from the original base artifact")
    if {o.point.point_id for o in observations} != set(artifact["training_point_ids"]) or len(observations) != len(
        artifact["training_point_ids"]
    ):
        raise ValueError("Residual fit must use exactly the base artifact training observations")
    spec = CompressorSpec(**artifact["compressor"])
    model = get_model(spec.compressor_type)
    points, residuals = defaultdict(list), defaultdict(list)
    for o in observations:
        prediction = model.predict(spec, o.point, artifact["parameters"])
        points[o.group_id].append(o.point)
        residuals[o.group_id].append([math.log(getattr(o, t) / prediction[t]) for t in TARGETS])
    if set(points) != set(artifact["training_group_ids"]):
        raise ValueError("Residual observations differ from base training groups")
    low, scale = normalizer([o.point for o in observations])
    centers = group_centers(points, low, scale)
    y = np.array([np.mean(residuals[g], axis=0) for g in points])
    coefficients = np.linalg.solve(rbf(centers, centers) + KERNEL_RIDGE * np.eye(len(centers)), y)
    corrected = deepcopy(artifact)
    corrected["prediction_extension"] = EXTENSION_VERSION
    corrected["residual_model"] = {
        "version": EXTENSION_VERSION,
        "length_scale": LENGTH_SCALE,
        "ridge": KERNEL_RIDGE,
        "input_low": low.tolist(),
        "input_scale": scale.tolist(),
        "centers": centers.tolist(),
        "group_ids": list(points),
        "coefficients": coefficients.tolist(),
        "targets": list(TARGETS),
        "semantics": "group-mean log(observed/base), one effective support per nominal group; no prediction-interval claim",
    }
    corrected["interval"] = {
        "status": "unavailable",
        "reason": "Base intervals invalidated by residual extension; needs new independent residual calibration",
    }
    corrected["assumptions"] = list(artifact.get("assumptions", [])) + [
        "Explicit statistical log-residual extension; base efficiency caps do not bound corrected predictions",
        "Fixed RBF prior hyperparameters, not instrument uncertainty; development CV is not new blind validation",
    ]
    validate_residual_model(corrected)
    return corrected


def validate_residual_model(artifact):
    extension = artifact.get("prediction_extension")
    residual = artifact.get("residual_model")
    if not extension and residual is None:
        return
    if extension != EXTENSION_VERSION or not isinstance(residual, dict) or residual.get("version") != EXTENSION_VERSION:
        raise ValueError("Unsupported/inconsistent residual prediction extension")
    if (
        residual.get("targets") != list(TARGETS)
        or residual.get("length_scale") != LENGTH_SCALE
        or residual.get("ridge") != KERNEL_RIDGE
    ):
        raise ValueError("Unsupported residual targets/hyperparameters; refit with a new version explicitly")
    ids = residual.get("group_ids", [])
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or set(ids) != set(artifact["training_group_ids"]):
        raise ValueError("Invalid residual training group mapping")
    for name, shape in (
        ("input_low", (4,)),
        ("input_scale", (4,)),
        ("centers", (len(ids), 4)),
        ("coefficients", (len(ids), 2)),
    ):
        try:
            array = np.asarray(residual.get(name), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid residual array: " + name) from exc
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Invalid residual array: " + name)
    if np.min(residual["input_scale"]) <= 0:
        raise ValueError("Invalid residual input scaling")
    from .kernels import training_groups

    groups = training_groups(artifact)
    low, scale = normalizer([p for ps in groups.values() for p in ps])
    centers = group_centers({g: groups[g] for g in ids}, low, scale)
    if (
        not np.allclose(low, residual["input_low"], rtol=0, atol=1e-12)
        or not np.allclose(scale, residual["input_scale"], rtol=0, atol=1e-12)
        or not np.allclose(centers, residual["centers"], rtol=0, atol=1e-12)
    ):
        raise ValueError("Residual centers/scaling differ from declared training inputs")


def apply_residual_correction(artifact, point, result):
    validate_residual_model(artifact)
    if not artifact.get("prediction_extension"):
        return result
    from .kernels import values

    residual = artifact["residual_model"]
    query = (values(point) - np.asarray(residual["input_low"])) / np.asarray(residual["input_scale"])
    delta = rbf(query[None, :], residual["centers"]) @ np.asarray(residual["coefficients"])
    result = dict(result)
    for t, correction in zip(TARGETS, delta[0]):
        try:
            corrected = result[t] * math.exp(float(correction))
        except OverflowError as exc:
            raise ValueError("Residual correction overflows") from exc
        if not math.isfinite(corrected) or corrected <= 0:
            raise ValueError("Residual prediction must be finite and positive")
        result["base_" + t] = result[t]
        result[t] = corrected
    for name in ("eta_v", "eta_is"):
        if name in result:
            result["base_" + name] = result.pop(name)
    result["prediction_extension"] = EXTENSION_VERSION
    return result
