"""Kernel/discrepancy tests are software checks, not real-machine validation."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from octwin.core import OperatingPoint, load_dataset
from octwin.demo import generate
from octwin.design import group_information
from octwin.development import run_development_study
from octwin.discrepancy import add_residual_correction
from octwin.kernels import kernel_selection_order, posterior_covariance, rank_kernel_groups, training_groups
from octwin.reporting import write_csv
from octwin.workflow import build_artifact, load_artifact, predict_point, recommend, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/public/arep21_r404a_65f"


@pytest.fixture(scope="module", params=["rolling_piston", "scroll"])
def specimen(tmp_path_factory, request):
    out = tmp_path_factory.mktemp(request.param)
    _, _, split, _ = generate(out, request.param)
    data = load_dataset(out / "data.csv", out / "metadata.json")
    # Explicit synthetic absolute-pressure discrepancy, never experimental data.
    obs = [
        replace(o, power_w=o.power_w * np.exp(0.06 * (o.point.suction_pressure_pa - 350000) / 100000))
        for o in data.observations
    ]
    data = replace(data, observations=obs)
    train = [o for o in obs if o.group_id in split["train"]]
    test = [o for o in obs if o.group_id in split["test"]]
    base = build_artifact(replace(data, observations=train), train)
    return data, train, test, base


def test_residual_smooth_counterexample_and_roundtrip(specimen, tmp_path):
    data, train, test, base = specimen
    original = deepcopy(base)
    base["interval"] = {"status": "available", "relative_radius": {"mass_flow_kg_s": 0.01, "power_w": 0.01}}
    corrected = add_residual_correction(base, train)
    assert corrected["interval"]["status"] == "unavailable"
    assert base["parameters"] == original["parameters"] and "residual_model" not in base
    write_json(tmp_path / "model.json", corrected)
    loaded = load_artifact(tmp_path / "model.json")
    before = [abs(predict_point(base, o.point)["power_w"] / o.power_w - 1) for o in test]
    after = [abs(predict_point(loaded, o.point)["power_w"] / o.power_w - 1) for o in test]
    assert np.mean(after) < np.mean(before)  # controlled L0 counterexample ONLY
    p = predict_point(loaded, test[0].point)
    assert p["power_w"] > 0 and "base_eta_v" in p and "eta_v" not in p
    assert p == predict_point(corrected, test[0].point)
    with pytest.raises(ValueError, match="residual extension"):
        recommend(corrected, [test[0].point], strategy="d_optimal")


@pytest.mark.parametrize("fault", ["ridge", "centers", "coefficients", "missing_flag", "unknown_extension"])
def test_invalid_residual_state_rejected(specimen, tmp_path, fault):
    _, train, _, base = specimen
    corrected = add_residual_correction(base, train)
    if fault == "ridge":
        corrected["residual_model"]["ridge"] = 0.02
    elif fault == "centers":
        corrected["residual_model"]["centers"][0][0] += 1
    elif fault == "coefficients":
        corrected["residual_model"]["coefficients"][0] = [0]
    elif fault == "missing_flag":
        corrected.pop("prediction_extension")
    else:
        corrected["prediction_extension"] = "unknown"
    write_json(tmp_path / "bad.json", corrected)
    with pytest.raises(ValueError, match="residual|Residual"):
        load_artifact(tmp_path / "bad.json")


def test_rbf_requires_matching_observations_and_resets_old_interval(specimen):
    _, train, _, base = specimen
    with pytest.raises(ValueError, match="exactly"):
        add_residual_correction(base, train[:-1])
    corrected = add_residual_correction(base, train)
    with pytest.raises(ValueError, match="already present"):
        add_residual_correction(corrected, train)


def test_kernel_distinguishes_equal_pressure_ratio_and_common_domain(specimen):
    _, _, test, base = specimen
    a = OperatingPoint("new-a", 300000, 1200000, 300, 3600)
    b = OperatingPoint("new-b", 400000, 1600000, 300, 3600)
    assert np.allclose(group_information(base, [a]), group_information(base, [b]))
    ranked = rank_kernel_groups(base, {"a": [a], "b": [b]})
    assert not np.isclose(
        ranked[0]["integrated_variance_gain"], ranked[1]["integrated_variance_gain"], rtol=0, atol=1e-9
    )
    result = recommend(base, [a, b], budget=2, strategy="kernel_ivr", costs={"new-a": 1, "new-b": 1000})
    assert len(result["selected"]) == 2 and result["selected"][0]["point_id"] == "new-a"
    outside = replace(a, point_id="outside", discharge_pressure_pa=3000000)
    result = recommend(base, [outside], strategy="kernel_ivr")
    assert not result["selected"] and result["rejected"]


def test_kernel_group_repeat_does_not_add_support(specimen):
    _, _, test, base = specimen
    repeated = deepcopy(base)
    repeated["training_inputs"] *= 2
    repeated["training_input_groups"] *= 2
    group = {"candidate": [test[0].point]}
    assert np.isclose(
        rank_kernel_groups(base, group)[0]["information_score"],
        rank_kernel_groups(repeated, group)[0]["information_score"],
        atol=1e-12,
    )
    assert len(training_groups(base)) == len(training_groups(repeated))


def test_variance_gain_equals_trace_reduction_and_is_positive():
    target = np.array([[0.0, 0.0], [1, 0], [0.5, 0.5], [0, 1]])
    cov = posterior_covariance(target, target[:1])
    i = 2
    column = cov[:, i].copy()
    gain = np.mean(column**2) / (cov[i, i] + 0.01)
    after = cov - np.outer(column, column) / (cov[i, i] + 0.01)
    assert gain > 0 and np.isclose(np.mean(np.diag(cov)) - np.mean(np.diag(after)), gain)
    assert np.linalg.eigvalsh(after).min() > -1e-10


@pytest.mark.parametrize("ridge", [0, -1, True, float("inf")])
def test_bad_kernel_regularization(ridge):
    with pytest.raises(ValueError, match="ridge"):
        posterior_covariance(np.ones((2, 4)), np.ones((1, 4)), ridge=ridge)


def test_kernel_only_inputs_horizon_and_no_duplicate_groups(specimen):
    data, train, test, base = specimen
    groups = {o.group_id: [o.point] for o in train + test}
    initial = [train[0].group_id, train[-1].group_id]
    trace = kernel_selection_order(groups, initial, 0, 5)
    assert len(trace["groups"]) == len(set(trace["groups"])) == 5
    assert len(trace["events"]) == 3
    with pytest.raises(ValueError, match="OperatingPoint"):
        rank_kernel_groups(base, {"bad": [test[0]]})
    with pytest.raises(ValueError, match="horizon"):
        kernel_selection_order(groups, initial, 0, 1)


def exporter():
    spec = importlib.util.spec_from_file_location("prepare_dev", ROOT / "tools/prepare_development40.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_development


@pytest.fixture(scope="module")
def development_files(tmp_path_factory):
    out = tmp_path_factory.mktemp("development40")
    config = exporter()(SOURCE / "data.csv", SOURCE / "metadata.json", SOURCE / "split.json", out)
    config["budgets"] = [9, 12]
    config["seeds"] = [0]
    return out, config


def test_export_excludes_labels_and_is_invariant_to_original_final_changes(tmp_path, development_files):
    folder, config = development_files
    # Preserve original CSV string rendering for unchanged rows.
    from octwin.core import read_rows

    raw = read_rows(SOURCE / "data.csv")
    for row in raw:
        if row["group_id"] in config["forbidden_groups"]:
            row["mass_flow_kg_s"] = str(float(row["mass_flow_kg_s"]) * 7)
            row["power_w"] = str(float(row["power_w"]) * 11)
    write_csv(tmp_path / "changed.csv", raw)
    exporter()(tmp_path / "changed.csv", SOURCE / "metadata.json", SOURCE / "split.json", tmp_path / "export")
    for name in ["data.csv", "metadata.json", "development_protocol.json"]:
        assert (folder / name).read_bytes() == (tmp_path / "export" / name).read_bytes()
    child = load_dataset(folder / "data.csv", folder / "metadata.json")
    assert len(child.observations) == 40
    assert not set(config["forbidden_groups"]).intersection(o.group_id for o in child.observations)


def test_original_full_dataset_rejected_by_development_api(development_files, tmp_path, monkeypatch):
    import octwin.development as module

    def forbidden_load(*args):
        raise AssertionError("Original CSV must not be opened by development loader")

    monkeypatch.setattr(module, "load_dataset", forbidden_load)
    _, config = development_files
    with pytest.raises(ValueError, match="development-only"):
        run_development_study(SOURCE / "data.csv", SOURCE / "metadata.json", config, tmp_path)
    assert not (tmp_path / "protocol.json").exists()


def test_development_cv_orders_all_frozen_before_scoring_and_immutable(development_files, tmp_path, monkeypatch):
    import octwin.development as module

    folder, config = development_files
    original = module.evaluate_fixed_orders

    def checked(data, pool, test, orders, budgets, **kw):
        saved = json.loads((tmp_path / "selection_traces.json").read_text())["traces"]
        assert len(saved) == 20 and all(len(t["groups"]) == 12 for t in saved)
        assert not {o.group_id for o in pool}.intersection(o.group_id for o in test)
        assert not set(config["forbidden_groups"]).intersection(o.group_id for o in pool + test)
        assert all(set(t["groups"]).issubset({o.group_id for o in pool}) for t in orders)
        return original(data, pool, test, orders, budgets, **kw)

    monkeypatch.setattr(module, "evaluate_fixed_orders", checked)
    result = run_development_study(folder / "data.csv", folder / "metadata.json", config, tmp_path)
    assert len(result["rows"]) == 160
    assert result["audit"]["original_calibration_or_final_groups_in_data"] == 0
    assert result["audit"]["unique_oof_groups"] == 31
    assert result["audit"]["max_acquired_groups"] == 12
    assert "DEVELOPMENT CROSS-VALIDATION ONLY" in (tmp_path / "report.html").read_text()
    changed = deepcopy(config)
    changed["budgets"] = [9]
    with pytest.raises(ValueError, match="Frozen development"):
        run_development_study(folder / "data.csv", folder / "metadata.json", changed, tmp_path)
    process = subprocess.run(
        [sys.executable, "-m", "octwin", "report", "--run", str(tmp_path), "--output", str(tmp_path / "rebuilt.html")],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_residual_and_kernel_cli(specimen, tmp_path):
    data, train, test, base = specimen
    write_json(tmp_path / "base.json", base)
    write_csv(tmp_path / "candidates.csv", [{**asdict(test[0].point), "relative_cost": 1}])
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "octwin",
            "recommend",
            "--model",
            str(tmp_path / "base.json"),
            "--candidates",
            str(tmp_path / "candidates.csv"),
            "--strategy",
            "kernel_ivr",
            "--cost-column",
            "relative_cost",
            "--approved",
            "--output",
            str(tmp_path / "recommendation.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads((tmp_path / "recommendation.json").read_text())["selected"]
    write_csv(
        tmp_path / "observed.csv",
        [
            {**asdict(o.point), "group_id": o.group_id, "mass_flow_kg_s": o.mass_flow_kg_s, "power_w": o.power_w}
            for o in train
        ],
    )
    write_json(tmp_path / "meta.json", data.metadata)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "octwin",
            "calibrate",
            str(tmp_path / "observed.csv"),
            "--meta",
            str(tmp_path / "meta.json"),
            "--residual-rbf",
            "--output",
            str(tmp_path / "cli-rbf.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert load_artifact(tmp_path / "cli-rbf.json")["prediction_extension"] == "rbf-log-residual-1"


def test_unidentifiable_base_retains_development_failures_without_fallback(specimen, tmp_path):
    from octwin.models import get_model

    data, _, _, base = specimen
    rows = []
    for i, ps in enumerate([250000, 350000, 450000]):
        for j, speed in enumerate([3000, 3600, 4200]):
            point = OperatingPoint(f"p-{i}-{j}", ps, 4 * ps, 300, speed)
            pred = get_model(data.spec.compressor_type).predict(data.spec, point, base["parameters"])
            rows.append(
                {
                    **asdict(point),
                    "group_id": point.point_id,
                    "mass_flow_kg_s": pred["mass_flow_kg_s"],
                    "power_w": pred["power_w"],
                }
            )
    write_csv(tmp_path / "constant-pr.csv", rows)
    metadata = {**data.metadata, "research_partition": "development_only"}
    write_json(tmp_path / "metadata.json", metadata)
    initial = ["p-0-0", "p-0-2", "p-2-0", "p-2-2"]
    groups = [r["group_id"] for r in rows]
    held = [g for g in groups if g not in initial]
    config = {
        "scope": "development_only",
        "development_groups": groups,
        "forbidden_groups": ["original-final"],
        "initial_groups": initial,
        "length_scale": 0.5,
        "kernel_ridge": 0.01,
        "strategies": ["random", "maximin", "d_optimal", "kernel_ivr"],
        "folds": [{"fold": i, "train": [g for g in groups if g != h], "validation": [h]} for i, h in enumerate(held)],
        "budgets": [4, 6],
        "seeds": [0],
    }
    result = run_development_study(tmp_path / "constant-pr.csv", tmp_path / "metadata.json", config, tmp_path / "run")
    assert result["audit"]["selector_stops"] == 5
    assert all(r["status"] == "failed_base_fit" for r in result["oof_predictions"] if r["family"] == "semiempirical")
    assert result["failures"]
    assert json.loads((tmp_path / "run/model_development_status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "run/model_development_base.json").exists()
    assert "SYNTHETIC SOFTWARE SELF-TEST ONLY" in (tmp_path / "run/report.html").read_text(encoding="utf-8")
