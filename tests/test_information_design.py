import copy
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from octwin.core import CompressorSpec, OperatingPoint, load_dataset
from octwin.design import (
    DEFAULT_RIDGE,
    group_information,
    information_gain,
    rank_information_groups,
    training_information,
)
from octwin.models import get_model
from octwin.reporting import write_csv
from octwin.workflow import build_artifact, predict_point, recommend, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public/arep21_r404a_65f"


@pytest.fixture(scope="module")
def case():
    data = load_dataset(DATA / "data.csv", DATA / "metadata.json")
    split = json.loads((DATA / "split.json").read_text())
    initial = split["mandatory_hull_training_groups"]
    observed = [o for o in data.observations if o.group_id in initial]
    artifact = build_artifact(replace(data, observations=observed), observed)
    candidates = [o.point for o in data.observations if o.group_id in split["train"] and o.group_id not in initial]
    return data, split, artifact, candidates


@pytest.mark.parametrize("kind", ["rolling_piston", "scroll"])
@pytest.mark.parametrize("pressure_ratio", [2.0, 4.0, 7.0])
def test_analytic_log_sensitivity_matches_finite_difference(kind, pressure_ratio):
    spec = CompressorSpec(kind, "SYNTHETIC", "derivative-test", "R134a", 9e-6)
    point = OperatingPoint("derivative", 300000, 300000 * pressure_ratio, 300, 3600)
    parameters = dict(v_intercept=2.0, v_loss=0.12, is_intercept=1.1, is_curvature=0.35)
    model = get_model(kind)
    analytic = model.log_output_jacobian(spec, point, parameters)
    numerical = np.empty((2, 4))
    for j, name in enumerate(model.parameter_names):
        plus, minus = parameters.copy(), parameters.copy()
        plus[name] += 1e-6
        minus[name] -= 1e-6
        a, b = model.predict(spec, point, plus), model.predict(spec, point, minus)
        numerical[:, j] = [(np.log(a[t]) - np.log(b[t])) / 2e-6 for t in ("mass_flow_kg_s", "power_w")]
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=2e-9)


def test_information_matrix_positive_and_repeats_do_not_inflate(case):
    _, _, artifact, _ = case
    fisher = training_information(artifact)
    np.testing.assert_allclose(fisher, fisher.T, atol=1e-12)
    assert np.linalg.eigvalsh(fisher).min() > 0
    repeated = copy.deepcopy(artifact)
    duplicate = dict(repeated["training_inputs"][0], point_id="new-repeat-id")
    repeated["training_inputs"].append(duplicate)
    repeated["training_input_groups"].append(repeated["training_input_groups"][0])
    repeated["training_point_ids"].append("new-repeat-id")
    np.testing.assert_allclose(training_information(repeated), fisher, rtol=1e-13, atol=1e-12)


def test_group_matrix_average_not_extra_independent_information(case):
    _, _, artifact, candidates = case
    point = candidates[0]
    np.testing.assert_allclose(
        group_information(artifact, [point, replace(point, point_id="repeat")]),
        group_information(artifact, [point]),
        atol=1e-12,
    )
    assert information_gain(training_information(artifact), group_information(artifact, [point])) > 0


def test_information_cost_ranking_and_inputs_only(case):
    data, _, artifact, candidates = case
    groups = {p.point_id: [p] for p in candidates[:6]}
    ranked = rank_information_groups(artifact, groups)
    assert all(r["logdet_gain"] > 0 and "power_w" not in r for r in ranked)
    costs = {g: 1.0 for g in groups}
    costs[ranked[0]["group_id"]] = 1e9
    changed = rank_information_groups(artifact, groups, costs)
    assert changed[0]["group_id"] != ranked[0]["group_id"]
    with pytest.raises(ValueError, match="OperatingPoint inputs only"):
        rank_information_groups(artifact, {"label-object": [data.observations[0]]})


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_invalid_information_ridge(case, bad):
    with pytest.raises(ValueError, match="ridge"):
        training_information(case[2], bad)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), True])
def test_invalid_relative_cost(case, bad):
    _, _, artifact, candidates = case
    with pytest.raises(ValueError, match="cost|Cost"):
        recommend(artifact, [candidates[0]], strategy="d_optimal", costs={candidates[0].point_id: bad})


def test_legacy_model_keeps_predict_coverage_but_requires_dopt_group_mapping(case):
    _, _, artifact, candidates = case
    legacy = copy.deepcopy(artifact)
    del legacy["training_input_groups"]
    assert "power_w" in predict_point(legacy, candidates[0])
    assert recommend(legacy, candidates[:3])["selected"]
    with pytest.raises(ValueError, match="training_input_groups"):
        recommend(legacy, candidates[:3], strategy="d_optimal")


def test_batch_information_recommendation_does_not_mutate_model_or_accept_duplicates(case):
    _, _, artifact, candidates = case
    before = copy.deepcopy(artifact)
    duplicate = replace(candidates[0], point_id="same-input-different-id")
    outside = replace(candidates[1], point_id="outside", suction_temperature_k=400)
    result = recommend(artifact, candidates[:5] + [duplicate, outside], 3, "d_optimal")
    assert len(result["selected"]) == 3
    assert len({p["point_id"] for p in result["selected"]}) == 3
    assert len(result["rejected"]) == 2
    assert artifact == before
    assert result["ridge"] == DEFAULT_RIDGE
    assert all(p["information_score"] == pytest.approx(p["logdet_gain"]) for p in result["selected"])
    with pytest.raises(ValueError, match="Costs must match"):
        recommend(artifact, candidates[:3], strategy="d_optimal", costs={})
    with pytest.raises(ValueError, match="only to d_optimal"):
        recommend(artifact, candidates[:3], costs={p.point_id: 1 for p in candidates[:3]})


def test_information_recommendation_cli_with_cost_column(tmp_path, case):
    _, _, artifact, candidates = case
    write_json(tmp_path / "model.json", artifact)
    write_csv(
        tmp_path / "candidates.csv", [{**asdict(p), "relative_cost": i + 1} for i, p in enumerate(candidates[:5])]
    )
    args = [
        sys.executable,
        "-m",
        "octwin",
        "recommend",
        "--model",
        str(tmp_path / "model.json"),
        "--candidates",
        str(tmp_path / "candidates.csv"),
        "--strategy",
        "d_optimal",
        "--cost-column",
        "relative_cost",
        "--approved",
        "--budget",
        "2",
        "--output",
        str(tmp_path / "recommendation.json"),
    ]
    process = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    result = json.loads((tmp_path / "recommendation.json").read_text())
    assert len(result["selected"]) == 2
    assert result["selected"][0]["relative_cost"] >= 1
