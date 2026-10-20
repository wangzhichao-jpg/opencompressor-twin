"""Four-parameter engineering-preview correlations, not geometry inversion.

m = rho_s * Vd * rpm/60 * eta_v; P = m * (h2s-hs)/eta_is.
The efficiency follows the declared target power boundary (electrical input
or compressor shaft); shaft power is never converted into electrical input.
eta_v = 1.25*sigmoid(v_intercept - v_loss*f(PR));
rolling piston: f=PR-1; scroll: f=ln(PR).
eta_is = 0.90*sigmoid(is_intercept - is_curvature*(ln(PR)-ln(3))**2).
These caps and PR=3 reference are explicit preview assumptions, not manufacturer
claims. Speed enters displacement throughput only; no speed-dependent loss model.
"""

import math

import numpy as np
from scipy.special import expit

from .properties import suction_properties

MODEL_VERSION = "semiempirical-preview-1"


class SemiEmpiricalModel:
    parameter_names = ("v_intercept", "v_loss", "is_intercept", "is_curvature")
    bounds = (np.array([-1.0, 0.0, -2.0, 0.0]), np.array([5.0, 1.0, 5.0, 2.0]))

    def __init__(self, compressor_type):
        self.compressor_type = compressor_type

    def predict(self, spec, point, parameters):
        if spec.compressor_type != self.compressor_type:
            raise ValueError("Model structure does not match compressor spec")
        if set(parameters) != set(self.parameter_names):
            raise ValueError(f"Parameters must be {self.parameter_names}")
        values = np.array([parameters[k] for k in self.parameter_names], dtype=float)
        if not np.isfinite(values).all() or np.any(values < self.bounds[0]) or np.any(values > self.bounds[1]):
            raise ValueError("Parameters outside preview bounds")
        a, b, c, d = values
        ratio = point.discharge_pressure_pa / point.suction_pressure_pa
        loss = ratio - 1 if self.compressor_type == "rolling_piston" else math.log(ratio)
        eta_v = float(1.25 * expit(a - b * loss))
        eta_is = float(0.90 * expit(c - d * (math.log(ratio) - math.log(3)) ** 2))
        props = suction_properties(point, spec.refrigerant)
        mass = props["density_kg_m3"] * spec.displacement_m3_rev * point.speed_rpm / 60 * eta_v
        power = mass * props["isentropic_work_j_kg"] / eta_is
        if not all(math.isfinite(v) and v > 0 for v in (mass, power, eta_v, eta_is)):
            raise ValueError("Invalid positive model output")
        return dict(mass_flow_kg_s=mass, power_w=power, eta_v=eta_v, eta_is=eta_is)

    def log_output_jacobian(self, spec, point, parameters):
        """d(log mass, log power)/d(parameters), NOT a noise/uncertainty model.

        Property/throughput terms do not depend on these four efficiency
        parameters. The current correlation's information depends on PR only;
        omitted absolute-pressure/thermal model discrepancy is not represented.
        """
        prediction = self.predict(spec, point, parameters)
        ratio = point.discharge_pressure_pa / point.suction_pressure_pa
        loss = ratio - 1 if self.compressor_type == "rolling_piston" else math.log(ratio)
        curvature = (math.log(ratio) - math.log(3)) ** 2
        dv = 1 - prediction["eta_v"] / 1.25
        di = 1 - prediction["eta_is"] / 0.90
        return np.array([[dv, -loss * dv, 0.0, 0.0], [dv, -loss * dv, -di, curvature * di]])


MODEL_REGISTRY = {kind: SemiEmpiricalModel(kind) for kind in ("rolling_piston", "scroll")}


def get_model(compressor_type):
    try:
        return MODEL_REGISTRY[compressor_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {compressor_type}") from exc
