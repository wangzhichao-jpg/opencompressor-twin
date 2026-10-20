"""CoolProp gas-only property boundary; no oil/two-phase suction model."""

import math
from functools import lru_cache

import CoolProp
from CoolProp.CoolProp import PhaseSI, PropsSI

PROPERTY_VERSION = CoolProp.__version__


@lru_cache(maxsize=8192)
def _properties(ps, pd, ts, fluid):
    try:
        phase = PhaseSI("P", ps, "T", ts, fluid)
        if phase not in {"gas", "supercritical_gas"}:
            raise ValueError(f"unsupported suction phase {phase}; require gas or supercritical_gas")
        rho = PropsSI("Dmass", "P", ps, "T", ts, fluid)
        h1 = PropsSI("Hmass", "P", ps, "T", ts, fluid)
        s1 = PropsSI("Smass", "P", ps, "T", ts, fluid)
        h2s = PropsSI("Hmass", "P", pd, "Smass", s1, fluid)
        work = h2s - h1
        if not all(math.isfinite(v) and v > 0 for v in (rho, work)):
            raise ValueError("invalid density or isentropic work")
        return rho, work
    except (ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"CoolProp: {exc}") from exc


def suction_properties(point, refrigerant):
    rho, work = _properties(
        point.suction_pressure_pa, point.discharge_pressure_pa, point.suction_temperature_k, refrigerant
    )
    return {"density_kg_m3": rho, "isentropic_work_j_kg": work}
