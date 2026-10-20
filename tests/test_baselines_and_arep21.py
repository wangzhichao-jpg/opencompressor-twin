import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from octwin.baselines import cubic_design, dew_features, fit_ahri10, predict_ahri10
from octwin.core import TARGETS, Observation, load_dataset
from octwin.reporting import write_csv
from octwin.workflow import benchmark, check_split, domain, linear_baseline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public/arep21_r404a_65f"


@pytest.fixture(scope="module")
def source():
    data = load_dataset(DATA / "data.csv", DATA / "metadata.json")
    split = json.loads((DATA / "split.json").read_text())
    train = [o for o in data.observations if o.group_id in split["train"]]
    test = [o for o in data.observations if o.group_id in split["test"]]
    return data, split, train, test


def test_real_source_units_and_independence(source):
    data, split, train, test = source
    check_split(data.observations, split)
    assert len(data.observations) == len({o.group_id for o in data.observations}) == 64
    assert (len(train), len(split["calibration"]), len(test)) == (40, 12, 12)
    assert data.metadata["source_kind"] == "experiment"
    assert data.metadata["power_boundary"] == "electrical_input"
    assert data.metadata["speed_definition"] == "nominal_mechanical_rpm"
    assert data.metadata["raw_sha256"] == "e2b1f34299e427a3927df6ddc6660253f1387e12e385c9fbab66a42a6743012f"
    assert data.spec.displacement_m3_rev == pytest.approx(3.11 * 0.000016387064)
    selected = next(o for o in data.observations if o.group_id == "A1-Td120-Te20")
    assert selected.mass_flow_kg_s == pytest.approx(484.5 * 0.45359237 / 3600)
    assert selected.power_w == 3278
    assert selected.point.suction_pressure_pa == pytest.approx(69 * 6894.757293168)
    assert selected.point.speed_rpm == pytest.approx(3497.7646302250805)
    assert selected.point.suction_temperature_k == pytest.approx((65 - 32) * 5 / 9 + 273.15)


def test_input_hull_split_is_scoreable_without_domain_relaxation(source):
    data, split, train, test = source
    from dataclasses import asdict

    artifact = {"training_inputs": [asdict(o.point) for o in train]}
    assert all(not domain(artifact, o.point)["status"].startswith("refused") for o in test)


def test_ahri_coefficients_recover_exact_cubic(source):
    data, _, train, test = source
    # Analytic target manufactured only for this numerical baseline test.
    center = np.array([265.0, 315.0])
    scale = np.array([25.0, 40.0])
    cm = np.array([0.05, 0.01, -0.005, 0.001, 0.002, 0.001, 0.0002, -0.0001, 0.0003, -0.0002])
    cp = np.array([2500.0, 100.0, 250.0, 20.0, 30.0, 10.0, 5.0, -3.0, 2.0, -1.0])

    def response(point):
        row = cubic_design(((dew_features(data.spec, point) - center) / scale)[None, :])[0]
        return float(row @ cm), float(row @ cp)

    perfect = [Observation(o.point, *response(o.point), o.group_id) for o in train]
    fitted = fit_ahri10(data.spec, perfect)
    assert fitted["diagnostics"]["rank"] == 10
    for o in test:
        predicted = predict_ahri10(data.spec, o.point, fitted)
        for target, value in zip(TARGETS, response(o.point)):
            assert predicted[target] == pytest.approx(value, rel=1e-10)


def test_ahri_no_source_polynomial_or_test_answers(source):
    data, _, train, test = source
    before = fit_ahri10(data.spec, train)
    changed_test = [replace(o, power_w=o.power_w * 100, mass_flow_kg_s=o.mass_flow_kg_s * 100) for o in test]
    # fit API has no parameter for held-out outputs or source polynomial coefficients.
    after = fit_ahri10(data.spec, train)
    assert before == after
    assert set(before["training_point_ids"]).isdisjoint(o.point.point_id for o in changed_test)
    assert all(
        predict_ahri10(data.spec, a.point, before) == predict_ahri10(data.spec, b.point, after)
        for a, b in zip(test, changed_test)
    )


def test_ahri_refuses_small_or_rank_deficient_data(source):
    data, _, train, _ = source
    with pytest.raises(ValueError, match=">10 independent"):
        fit_ahri10(data.spec, train[:10])
    repeated = [replace(o, group_id="one-repeated-condition") for o in train]
    with pytest.raises(ValueError, match="independent"):
        fit_ahri10(data.spec, repeated)
    two_levels = [o for o in data.observations if o.group_id.startswith(("A1-Td140-", "A1-Td130-"))]
    with pytest.raises(ValueError, match="rank"):
        fit_ahri10(data.spec, two_levels)


@pytest.mark.parametrize("field", ["speed_rpm", "suction_temperature_k"])
def test_ahri_refuses_mixed_regimes(source, field):
    data, _, train, test = source
    changed = [replace(train[0], point=replace(train[0].point, **{field: getattr(train[0].point, field) + 1}))] + train[
        1:
    ]
    with pytest.raises(ValueError, match="fixed"):
        fit_ahri10(data.spec, changed)
    fitted = fit_ahri10(data.spec, train)
    with pytest.raises(ValueError, match="regime"):
        predict_ahri10(data.spec, replace(test[0].point, **{field: getattr(test[0].point, field) + 1}), fitted)


def test_affine_fixed_column_rank_is_not_inflated(source):
    _, _, train, test = source
    status = [{"point_id": o.point.point_id, "status": "in_training_hull"} for o in test]
    result = linear_baseline(train, test, status)
    # Only ps and pd vary: intercept + two input axes, not an extra floating residual axis.
    assert result["rank"] == 3
    assert result["metrics"]["predicted"] == len(test)


@pytest.mark.parametrize("change", ["missing", "bad_rate", "variable_rpm"])
def test_nominal_speed_requires_traceable_fixed_rate(tmp_path, source, change):
    data, _, _, _ = source
    meta = copy.deepcopy(data.metadata)
    observations = data.observations
    if change == "missing":
        del meta["nominal_speed_provenance"]
    elif change == "bad_rate":
        meta["nominal_speed_provenance"]["volume_rate_m3_s"] *= 2
    else:
        observations = [replace(observations[0], point=replace(observations[0].point, speed_rpm=3600))] + observations[
            1:
        ]
    from dataclasses import asdict

    rows = [
        {**asdict(o.point), "group_id": o.group_id, "mass_flow_kg_s": o.mass_flow_kg_s, "power_w": o.power_w}
        for o in observations
    ]
    write_csv(tmp_path / "data.csv", rows)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="[Nn]ominal|nominal"):
        load_dataset(tmp_path / "data.csv", tmp_path / "meta.json")


def test_real_benchmark_records_all_three_families_and_scope(tmp_path, source):
    data, split, _, _ = source
    result = benchmark(DATA / "data.csv", DATA / "metadata.json", tmp_path, split)
    for score in (result["metrics"], result["baseline"]["metrics"], result["ahri10_baseline"]["metrics"]):
        assert score["total"] == score["predicted"] == 12
        for target in TARGETS:
            assert np.isfinite(score[target]["mare_percent"])
    assert result["ahri10_baseline"]["diagnostics"]["rank"] == 10
    assert result["baseline"]["rank"] == 3
    assert result["interval"]["groups"] == 12
    assert result["source_review"]["speed_definition"] == "nominal_mechanical_rpm"
    assert (tmp_path / "ahri10_model.json").is_file()
    assert (tmp_path / "ahri10_predictions.csv").is_file()
    assert "AHRI-style" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert result["stratified_metrics"]["suction_pressure_pa"]["outside_training_range"] == 0


def test_completed_run_cannot_replace_its_frozen_split(tmp_path, source):
    _, split, _, _ = source
    benchmark(DATA / "data.csv", DATA / "metadata.json", tmp_path, split)
    before = (tmp_path / "result.json").read_bytes()
    altered = copy.deepcopy(split)
    altered["test"][0], altered["calibration"][0] = altered["calibration"][0], altered["test"][0]
    with pytest.raises(ValueError, match="different dataset/split"):
        benchmark(DATA / "data.csv", DATA / "metadata.json", tmp_path, altered)
    assert (tmp_path / "result.json").read_bytes() == before


def test_explicit_empty_calibration_split_is_not_ignored(tmp_path):
    (tmp_path / "bad.json").write_text("{}")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "octwin",
            "calibrate",
            str(DATA / "data.csv"),
            "--meta",
            str(DATA / "metadata.json"),
            "--split",
            str(tmp_path / "bad.json"),
            "--output",
            str(tmp_path / "not-a-model.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert not (tmp_path / "not-a-model.json").exists()


def test_ahri_roundtrip_and_property_version_guard(source):
    data, _, train, test = source
    fitted = fit_ahri10(data.spec, train)
    loaded = json.loads(json.dumps(fitted))
    assert predict_ahri10(data.spec, test[0].point, fitted) == predict_ahri10(data.spec, test[0].point, loaded)
    loaded["property_version"] = "incompatible"
    with pytest.raises(ValueError, match="property version"):
        predict_ahri10(data.spec, test[0].point, loaded)
