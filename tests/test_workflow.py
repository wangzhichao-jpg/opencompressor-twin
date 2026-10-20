import copy
import subprocess
import sys
from dataclasses import replace

import pytest

from octwin.core import OperatingPoint, load_dataset
from octwin.demo import generate
from octwin.workflow import (
    benchmark,
    build_artifact,
    calibrate_interval,
    check_split,
    domain,
    load_artifact,
    metrics,
    predict_point,
    recommend,
    write_json,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("workflow")
    _, _, split, candidates = generate(folder, "scroll")
    result = benchmark(folder / "data.csv", folder / "metadata.json", folder / "run", split)
    artifact = load_artifact(folder / "run" / "model.json")
    dataset = load_dataset(folder / "data.csv", folder / "metadata.json")
    return folder, split, candidates, result, artifact, dataset


def test_roundtrip_and_metrics(run):
    folder, split, candidates, result, artifact, dataset = run
    assert result["metrics"]["total"] == 15
    assert result["metrics"]["predicted"] == 15
    assert result["interval"]["status"] == "available"
    assert "SOFTWARE SELF-TEST" in result["evidence_label"]
    assert set(artifact["training_group_ids"]).isdisjoint(split["test"])
    assert set(artifact["training_group_ids"]).isdisjoint(split["calibration"])
    first = OperatingPoint(**candidates[0])
    before = predict_point(artifact, first)
    after = predict_point(load_artifact(folder / "run" / "model.json"), first)
    assert before == after
    assert (folder / "run" / "predictions.csv").is_file()
    html = (folder / "run" / "report.html").read_text()
    assert "SOFTWARE SELF-TEST" in html and "<svg" in html


def test_extrapolation_refusal_and_no_fake_capacity(run):
    _, _, candidates, _, artifact, _ = run
    point = OperatingPoint(**candidates[0])
    assert "reference_capacity_w" not in predict_point(artifact, point)
    outside = replace(point, suction_pressure_pa=200000)
    result = predict_point(artifact, outside)
    assert result["status"] == "refused_extrapolation"
    assert "power_w" not in result


def test_hull_not_only_bounding_box():
    points = [
        OperatingPoint(str(i), ps, pd, 300, 3600)
        for i, (ps, pd) in enumerate([(300000, 1000000), (400000, 1000000), (300000, 2000000)])
    ]
    from dataclasses import asdict

    artifact = {"training_inputs": [asdict(p) for p in points]}
    assert (
        domain(artifact, OperatingPoint("outside", 390000, 1900000, 300, 3600))["status"]
        == "refused_outside_training_hull"
    )
    assert domain(artifact, OperatingPoint("inside", 320000, 1200000, 300, 3600))["status"] == "in_training_hull"
    assert domain(artifact, OperatingPoint("speed", 320000, 1200000, 300, 4000))["status"] == "refused_extrapolation"


def test_group_leakage_rejected(run):
    _, split, _, _, _, data = run
    broken = copy.deepcopy(split)
    broken["test"].append(broken["train"][0])
    with pytest.raises(ValueError, match="overlap"):
        check_split(data.observations, broken)
    broken = copy.deepcopy(split)
    broken["test"].pop()
    with pytest.raises(ValueError, match="partition"):
        check_split(data.observations, broken)


def test_no_test_output_leakage(run, tmp_path):
    folder, split, candidates, result, artifact, dataset = run
    # Change only final-test answers; trained coefficients and calibrated intervals must not change.
    changed = [
        replace(o, mass_flow_kg_s=o.mass_flow_kg_s * 10, power_w=o.power_w * 20) if o.group_id in split["test"] else o
        for o in dataset.observations
    ]
    new_data = replace(dataset, observations=changed)
    train = [o for o in changed if o.group_id in split["train"]]
    cal = [o for o in changed if o.group_id in split["calibration"]]
    other = build_artifact(new_data, train)
    other["interval"] = calibrate_interval(other, cal)
    assert artifact["parameters"] == other["parameters"]
    assert artifact["interval"] == other["interval"]
    points = [OperatingPoint(**c) for c in candidates]
    assert recommend(artifact, points) == recommend(other, points)


def test_interval_insufficiency_and_overlap(run):
    _, split, _, _, artifact, dataset = run
    cal = [o for o in dataset.observations if o.group_id in split["calibration"]]
    assert calibrate_interval(artifact, cal[:2])["status"] == "unavailable"
    train = [o for o in dataset.observations if o.group_id in split["train"]]
    with pytest.raises(ValueError, match="overlap"):
        calibrate_interval(artifact, train)
    with pytest.raises(ValueError, match="coverage"):
        calibrate_interval(artifact, cal, 1.0)


def test_recommend_inputs_only_and_reject_seen(run):
    _, _, candidates, _, artifact, _ = run
    points = [OperatingPoint(**p) for p in candidates]
    chosen = recommend(artifact, points, 2)
    assert len(chosen["selected"]) == 2
    assert chosen["selected"][0]["point_id"] != chosen["selected"][1]["point_id"]
    assert "power_w" not in chosen["selected"][0]
    observed = OperatingPoint(**artifact["training_inputs"][0])
    assert recommend(artifact, [observed])["selected"] == []
    with pytest.raises(ValueError):
        recommend(artifact, points, 0)


def test_metrics_keep_failures(run):
    _, _, _, _, artifact, data = run
    o = data.observations[0]
    result = metrics([o, o], [{"status": "refused"}, {"mass_flow_kg_s": o.mass_flow_kg_s, "power_w": o.power_w}])
    assert result["total"] == 2 and result["refused_or_failed"] == 1
    assert result["power_w"]["mare_percent"] == 0


def test_artifact_version_rejection(run, tmp_path):
    _, _, _, _, artifact, _ = run
    changed = copy.deepcopy(artifact)
    changed["environment"]["CoolProp"] = "incompatible"
    write_json(tmp_path / "bad.json", changed)
    with pytest.raises(ValueError, match="CoolProp version"):
        load_artifact(tmp_path / "bad.json")


@pytest.mark.parametrize("command", ["validate", "calibrate", "predict", "recommend", "report"])
def test_cli_end_to_end(run, tmp_path, command):
    folder, split, candidates, result, artifact, data = run
    args = [sys.executable, "-m", "octwin", command]
    if command == "validate":
        args += [str(folder / "data.csv"), "--meta", str(folder / "metadata.json")]
    elif command == "calibrate":
        args += [
            str(folder / "data.csv"),
            "--meta",
            str(folder / "metadata.json"),
            "--split",
            str(folder / "split.json"),
            "--output",
            str(tmp_path / "model.json"),
        ]
    elif command in {"predict", "recommend"}:
        args += [
            "--model",
            str(folder / "run" / "model.json"),
            "--conditions" if command == "predict" else "--candidates",
            str(folder / "candidates.csv"),
            "--output",
            str(tmp_path / ("out.csv" if command == "predict" else "out.json")),
        ]
        if command == "recommend":
            args += ["--approved"]
    else:
        args += ["--run", str(folder / "run"), "--output", str(tmp_path / "report.html")]
    process = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr


def test_recommend_cli_rejects_target_columns(run, tmp_path):
    folder, _, _, _, _, _ = run
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "octwin",
            "recommend",
            "--model",
            str(folder / "run" / "model.json"),
            "--candidates",
            str(folder / "data.csv"),
            "--approved",
            "--output",
            str(tmp_path / "bad.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "input-only" in process.stderr


def test_html_escaping(run, tmp_path):
    from octwin.reporting import report

    result = copy.deepcopy(run[3])
    result["source"] = '<script>alert("x")</script>'
    report(result, tmp_path / "report.html")
    page = (tmp_path / "report.html").read_text()
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
