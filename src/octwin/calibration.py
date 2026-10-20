"""Bounded relative least squares. Weights are NOT measurement uncertainty."""

import numpy as np
from scipy.optimize import least_squares

from .models import get_model


def calibrate(spec, observations):
    if len({o.group_id for o in observations}) < 4:
        raise ValueError("Calibration requires at least four independent condition groups")
    if len({o.point.point_id for o in observations}) != len(observations):
        raise ValueError("Duplicate training point_id")
    model = get_model(spec.compressor_type)
    names = model.parameter_names
    # Equal group influence prevents long repeat series dominating a nominal condition.
    counts = {g: sum(o.group_id == g for o in observations) for g in {o.group_id for o in observations}}

    def residual(values):
        params = dict(zip(names, values))
        result = []
        for o in observations:
            p = model.predict(spec, o.point, params)
            weight = counts[o.group_id] ** -0.5
            result.extend(
                ((p["mass_flow_kg_s"] / o.mass_flow_kg_s - 1) * weight, (p["power_w"] / o.power_w - 1) * weight)
            )
        return np.asarray(result)

    fits = [
        least_squares(residual, start, bounds=model.bounds, max_nfev=2000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        for start in ([1.5, 0.1, 1.0, 0.3], [0.5, 0.3, 0.5, 0.7], [3.0, 0.05, 2.0, 0.1])
    ]
    good = [fit for fit in fits if fit.success and np.isfinite(fit.cost)]
    if not good:
        raise ValueError("Calibration optimizer failed for all starts")
    fit = min(good, key=lambda f: f.cost)
    # Scale columns by declared parameter range before assessing local identifiability.
    jacobian = fit.jac * (model.bounds[1] - model.bounds[0])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian, tol=singular[0] * 1e-8))
    if rank < len(names):
        raise ValueError(f"Unidentifiable calibration: scaled Jacobian rank {rank}/{len(names)}")
    condition = float(singular[0] / singular[-1])
    warnings = ["Local rank does not prove global uniqueness; no physical gap inference"]
    if condition > 1e5:
        warnings.append("Ill-conditioned local parameter estimates")
    active = [names[i] for i, v in enumerate(fit.active_mask) if v]
    if active:
        warnings.append("Parameter bounds touched: " + ", ".join(active))
    return {
        "parameters": dict(zip(names, map(float, fit.x))),
        "diagnostics": {
            "rank": rank,
            "condition_number": condition,
            "singular_values": singular.tolist(),
            "warning": warnings,
            "relative_residual_sum_squares": float(2 * fit.cost),
            "weighting": "relative residuals, equal total weight per repeat group; not metrological uncertainty",
            "successful_starts": len(good),
            "multi_start_parameter_spread": np.ptp([f.x for f in good], axis=0).tolist(),
        },
        "training_point_ids": [o.point.point_id for o in observations],
    }
