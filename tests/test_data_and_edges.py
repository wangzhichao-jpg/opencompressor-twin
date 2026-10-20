import json
from dataclasses import replace
from pathlib import Path

import pytest

from octwin.core import OperatingPoint, load_dataset
from octwin.demo import generate
from octwin.workflow import benchmark, build_artifact, metrics, predict_point, recommend

ROOT = Path(__file__).resolve().parents[1]


def test_public_subset_source_boundary_and_frozen_split():
    directory = ROOT / "data/public/sanden_tgve08_dry"
    data = load_dataset(directory / "data.csv", directory / "metadata.json")
    split = json.loads((directory / "split.json").read_text())
    assert len(data.observations) == 15
    assert len({o.group_id for o in data.observations}) == 12
    assert len(split["test"]) == 2
    assert split["calibration"] == []
    assert data.metadata["admission_status"] == "exploratory_only"
    assert "NOT_electrical" in data.metadata["power_boundary"]
    assert data.spec.displacement_m3_rev == pytest.approx(86e-6)
    assert data.metadata["license"].startswith("CC BY 4.0")
    assert data.fingerprint == "92bd1c189070f02b0be99bd8f873ee634a0b6b4b8177b8d33b93d478c65da070"


@pytest.fixture
def ready(tmp_path):
    _, _, split, candidates = generate(tmp_path, "rolling_piston")
    data = load_dataset(tmp_path / "data.csv", tmp_path / "metadata.json")
    train = [o for o in data.observations if o.group_id in split["train"]]
    artifact = build_artifact(data, train)
    return tmp_path, data, train, artifact, candidates


def test_capacity_boundary_not_silently_electrical_cop(ready):
    _, _, _, artifact, candidates = ready
    artifact["metadata"]["reference_enthalpy_difference_j_kg"] = 150000
    artifact["metadata"]["capacity_reference"] = "explicit test reference, not actual suction enthalpy"
    artifact["metadata"]["power_boundary"] = "compressor_shaft"
    result = predict_point(artifact, OperatingPoint(**candidates[0]))
    assert "reference_capacity_w" in result
    assert "reference_capacity_to_power_ratio" in result
    assert "reference_cop" not in result
    artifact["metadata"]["power_boundary"] = "electrical_input"
    assert "reference_cop" in predict_point(artifact, OperatingPoint(**candidates[0]))


def test_no_foreign_training_observations(ready):
    _, data, train, _, _ = ready
    changed = [replace(train[0], power_w=train[0].power_w * 2)] + train[1:]
    with pytest.raises(ValueError, match="unchanged subset"):
        build_artifact(data, changed)


def test_no_implicit_fallback_for_empty_split(ready):
    folder, _, _, _, _ = ready
    with pytest.raises(ValueError, match="Split requires"):
        benchmark(folder / "data.csv", folder / "metadata.json", folder / "run", split={})


def test_metrics_length_must_match(ready):
    _, data, _, _, _ = ready
    with pytest.raises(ValueError, match="One prediction"):
        metrics(data.observations, [])


def test_duplicate_candidates_and_boolean_budget(ready):
    _, _, _, artifact, candidates = ready
    point = OperatingPoint(**candidates[0])
    with pytest.raises(ValueError, match="Duplicate"):
        recommend(artifact, [point, point])
    with pytest.raises(ValueError, match="positive integer"):
        recommend(artifact, [point], True)
