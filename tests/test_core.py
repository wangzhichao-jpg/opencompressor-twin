import json
from dataclasses import replace

import numpy as np
import pytest

from octwin.calibration import calibrate
from octwin.core import CompressorSpec, Observation, OperatingPoint, load_dataset
from octwin.demo import generate
from octwin.models import get_model
from octwin.properties import suction_properties


@pytest.fixture
def spec():
    return CompressorSpec("rolling_piston", "test", "test", "R134a", 9e-6)


@pytest.fixture
def params():
    return dict(v_intercept=2.0, v_loss=0.12, is_intercept=1.1, is_curvature=0.35)


@pytest.mark.parametrize(
    "field,value",
    [
        ("speed_rpm", 0),
        ("suction_pressure_pa", -2),
        ("suction_temperature_k", float("nan")),
        ("discharge_pressure_pa", float("inf")),
    ],
)
def test_invalid_si(field, value):
    row = dict(
        point_id="x",
        suction_pressure_pa=300000,
        discharge_pressure_pa=1200000,
        suction_temperature_k=300,
        speed_rpm=3600,
    )
    row[field] = value
    with pytest.raises(ValueError):
        OperatingPoint(**row)


def test_pressure_order():
    with pytest.raises(ValueError):
        OperatingPoint("x", 300000, 250000, 300, 3600)


@pytest.mark.parametrize("kind", ["rolling_piston", "scroll"])
def test_model_scaling_and_parameters(spec, params, kind):
    spec = replace(spec, compressor_type=kind)
    p = OperatingPoint("x", 300000, 1200000, 300, 3600)
    model = get_model(kind)
    result = model.predict(spec, p, params)
    double = model.predict(spec, replace(p, speed_rpm=7200), params)
    assert double["mass_flow_kg_s"] == pytest.approx(2 * result["mass_flow_kg_s"])
    assert double["power_w"] == pytest.approx(2 * result["power_w"])
    assert 0 < result["eta_is"] < 0.9
    assert 0 < result["eta_v"] < 1.25
    assert result["power_w"] * result["eta_is"] == pytest.approx(
        result["mass_flow_kg_s"] * suction_properties(p, "R134a")["isentropic_work_j_kg"]
    )
    with pytest.raises(ValueError):
        model.predict(spec, p, {**params, "v_loss": float("nan")})
    with pytest.raises(ValueError):
        model.predict(
            replace(spec, compressor_type="scroll" if kind == "rolling_piston" else "rolling_piston"), p, params
        )


def test_properties_reject_liquid_and_unknown():
    with pytest.raises(ValueError, match="phase"):
        suction_properties(OperatingPoint("liquid", 1000000, 2000000, 280, 3000), "R134a")
    with pytest.raises(ValueError, match="CoolProp"):
        suction_properties(OperatingPoint("gas", 300000, 1200000, 300, 3000), "not-a-real-fluid")


def test_spec():
    with pytest.raises(ValueError):
        CompressorSpec("reciprocating", "a", "b", "R134a", 1e-5)
    with pytest.raises(ValueError):
        CompressorSpec("scroll", "", "b", "R134a", 1e-5)


@pytest.mark.parametrize("kind", ["rolling_piston", "scroll"])
def test_parameter_recovery(spec, params, kind):
    spec = replace(spec, compressor_type=kind)
    model = get_model(kind)
    observed = []
    for i, pr in enumerate(np.linspace(2, 7, 10)):
        p = OperatingPoint(str(i), 300000, 300000 * pr, 300, 3600)
        prediction = model.predict(spec, p, params)
        observed.append(Observation(p, prediction["mass_flow_kg_s"], prediction["power_w"], str(i)))
    fit = calibrate(spec, observed)
    assert fit["diagnostics"]["rank"] == 4
    for k in params:
        assert fit["parameters"][k] == pytest.approx(params[k], rel=1e-5)


def test_insufficient_and_unidentifiable(spec, params):
    p = OperatingPoint("x", 300000, 1200000, 300, 3600)
    pred = get_model(spec.compressor_type).predict(spec, p, params)
    obs = [Observation(replace(p, point_id=str(i)), pred["mass_flow_kg_s"], pred["power_w"], str(i)) for i in range(8)]
    with pytest.raises(ValueError, match="four independent"):
        calibrate(spec, obs[:3])
    with pytest.raises(ValueError, match="rank"):
        calibrate(spec, obs)


def test_loading_fingerprint_and_provenance(tmp_path):
    generate(tmp_path, "scroll")
    data = load_dataset(tmp_path / "data.csv", tmp_path / "metadata.json")
    assert len(data.observations) == 81
    assert len(data.fingerprint) == 64
    meta = data.metadata
    meta["validation_role"] = "public_validation"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Synthetic"):
        load_dataset(tmp_path / "data.csv", tmp_path / "metadata.json")


@pytest.mark.parametrize(
    "key,value", [("pressure_basis", "gauge"), ("speed_definition", "electrical_hz"), ("power_boundary", "")]
)
def test_metadata_boundaries(tmp_path, key, value):
    generate(tmp_path, "scroll")
    path = tmp_path / "metadata.json"
    meta = json.loads(path.read_text())
    meta[key] = value
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "data.csv", path)


def test_duplicate_input_groups(tmp_path):
    from octwin.reporting import write_csv

    rows, meta, _, _ = generate(tmp_path, "scroll")
    duplicate = {**rows[0], "point_id": "duplicate", "group_id": "different-group"}
    write_csv(tmp_path / "data.csv", rows + [duplicate])
    with pytest.raises(ValueError, match="Identical conditions"):
        load_dataset(tmp_path / "data.csv", tmp_path / "metadata.json")
