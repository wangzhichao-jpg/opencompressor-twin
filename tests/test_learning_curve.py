import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from octwin.core import load_dataset
from octwin.evaluation import input_selection_order, run_learning_curve
from octwin.reporting import write_csv
from octwin.workflow import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public/arep21_r404a_65f"


def material():
    data = load_dataset(DATA / "data.csv", DATA / "metadata.json")
    split = json.loads((DATA / "split.json").read_text())
    groups = {g: [o.point for o in data.observations if o.group_id == g] for g in split["train"]}
    return data, split, groups


@pytest.mark.parametrize("strategy", ["random", "maximin"])
def test_selector_inputs_only_order_and_group_isolation(strategy):
    data, split, groups = material()
    initial = split["mandatory_hull_training_groups"]
    order = input_selection_order(groups, initial, strategy, 0)
    assert order[: len(initial)] == initial
    assert len(order) == len(set(order)) == 40
    assert set(order) == set(split["train"])
    assert set(order).isdisjoint(split["calibration"] + split["test"])
    bad = {g: [o for o in data.observations if o.group_id == g] for g in split["train"]}
    with pytest.raises(ValueError, match="OperatingPoint inputs only"):
        input_selection_order(bad, initial, strategy, 0)


def test_curve_retains_ineligible_baseline_and_counts_initial_cost(tmp_path):
    _, split, _ = material()
    result = run_learning_curve(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (9, 12), (0, 1))
    assert len(result["rows"]) == 24
    assert len(result["failures"]) == 4
    assert all(r["family"] == "ahri10" and r["budget_groups"] == 9 for r in result["failures"])
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["initial_group_count"] == 9
    assert all(len(r["groups"]) == 40 for r in protocol["orders"])
    assert (tmp_path / "report.html").is_file()
    assert "SECONDARY RETROSPECTIVE" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert all(r["test_total"] == 12 for r in result["rows"])
    assert all(r["predicted"] == 12 for r in result["rows"] if r["status"] == "available")


def test_unrevealed_and_final_outputs_never_change_selection_or_initial_fit(tmp_path):
    data, split, _ = material()
    original = run_learning_curve(DATA / "data.csv", DATA / "metadata.json", split, tmp_path / "original", (9,), (0,))
    revealed = set(split["mandatory_hull_training_groups"])

    def write_modified(directory, scale_final):
        rows = []
        for o in data.observations:
            # Reserve final outputs in the first check; modify them only in the second.
            scale = 100 if o.group_id not in revealed and (scale_final or o.group_id not in split["test"]) else 1
            rows.append(
                {
                    **asdict(o.point),
                    "group_id": o.group_id,
                    "mass_flow_kg_s": o.mass_flow_kg_s * scale,
                    "power_w": o.power_w * scale,
                }
            )
        write_csv(directory / "data.csv", rows)
        write_json(directory / "meta.json", data.metadata)
        return run_learning_curve(directory / "data.csv", directory / "meta.json", split, directory / "run", (9,), (0,))

    unseen_changed = write_modified(tmp_path / "unseen", False)
    assert original["rows"] == unseen_changed["rows"]
    final_changed = write_modified(tmp_path / "final", True)
    paths = [
        tmp_path / "original/protocol.json",
        tmp_path / "unseen/run/protocol.json",
        tmp_path / "final/run/protocol.json",
    ]
    protocols = [json.loads(p.read_text()) for p in paths]
    assert protocols[0]["orders"] == protocols[1]["orders"] == protocols[2]["orders"]
    assert original["rows"][0]["mass_flow_kg_s_mare_percent"] != final_changed["rows"][0]["mass_flow_kg_s_mare_percent"]


@pytest.mark.parametrize("budgets", [(), (8,), (41,), (9, 9), (True,)])
def test_invalid_curve_budget(tmp_path, budgets):
    _, split, _ = material()
    with pytest.raises(ValueError, match="Budgets"):
        run_learning_curve(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, budgets, (0,))


def test_curve_protocol_cannot_be_silently_replaced(tmp_path):
    _, split, _ = material()
    run_learning_curve(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (9,), (0,))
    before = (tmp_path / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="protocol differs"):
        run_learning_curve(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (12,), (0,))
    assert (tmp_path / "protocol.json").read_bytes() == before


def test_learning_curve_and_report_cli(tmp_path):
    command = [
        sys.executable,
        "-m",
        "octwin",
        "learning-curve",
        str(DATA / "data.csv"),
        "--meta",
        str(DATA / "metadata.json"),
        "--split",
        str(DATA / "split.json"),
        "--budgets",
        "9,12",
        "--seeds",
        "0",
        "--output",
        str(tmp_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    process = subprocess.run(
        [sys.executable, "-m", "octwin", "report", "--run", str(tmp_path), "--output", str(tmp_path / "rebuilt.html")],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert process.returncode == 0, process.stderr
    assert "fixed-budget learning curves" in (tmp_path / "rebuilt.html").read_text(encoding="utf-8")
