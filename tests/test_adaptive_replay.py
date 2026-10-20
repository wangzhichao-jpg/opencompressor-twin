import copy
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from octwin.core import load_dataset
from octwin.replay import TrainingPoolOracle, adaptive_information_order, run_adaptive_replay
from octwin.reporting import write_csv
from octwin.workflow import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public/arep21_r404a_65f"


def material():
    data = load_dataset(DATA / "data.csv", DATA / "metadata.json")
    split = json.loads((DATA / "split.json").read_text())
    pool = [o for o in data.observations if o.group_id in split["train"]]
    groups = {g: [o.point for o in pool if o.group_id == g] for g in split["train"]}
    return data, split, pool, groups


@pytest.fixture(scope="module")
def adaptive_trace():
    data, split, pool, groups = material()
    oracle = TrainingPoolOracle(pool)
    trace = adaptive_information_order(
        data.spec, data.metadata, data.fingerprint, oracle, groups, split["mandatory_hull_training_groups"], 0
    )
    return data, split, pool, groups, trace


def test_oracle_holds_training_pool_only_and_reveals_once():
    _, split, pool, _ = material()
    oracle = TrainingPoolOracle(pool)
    assert oracle.observations() == []
    with pytest.raises(ValueError, match="approved training pool"):
        oracle.reveal(split["test"][0])
    oracle.reveal(split["train"][0])
    assert len(oracle.observations()) == 1
    with pytest.raises(ValueError, match="already revealed"):
        oracle.reveal(split["train"][0])


def test_real_order_reveals_after_selection_and_refits_each_step(adaptive_trace):
    _, split, _, _, trace = adaptive_trace
    initial = split["mandatory_hull_training_groups"]
    assert trace["status"] == "completed"
    assert len(trace["groups"]) == len(set(trace["groups"])) == 40
    assert len(trace["events"]) == trace["fit_attempts"] == 31
    assert trace["reveal_log"] == trace["groups"]
    assert set(trace["groups"]).isdisjoint(split["calibration"] + split["test"])
    for i, event in enumerate(trace["events"]):
        before = trace["groups"][: len(initial) + i]
        assert event["observed_groups_before_selection"] == before
        assert event["selected_group_id"] not in before
        assert event["selected_group_id"] == trace["groups"][len(before)]
        assert event["local_fit_rank"] == 4
    assert trace["events"][0]["parameters_before_selection"] != trace["events"][1]["parameters_before_selection"]


def test_unrevealed_answers_cannot_choose_the_first_new_point(adaptive_trace):
    data, split, pool, groups, original = adaptive_trace
    initial = set(split["mandatory_hull_training_groups"])
    changed = [
        replace(o, mass_flow_kg_s=o.mass_flow_kg_s * 1.3, power_w=o.power_w * 1.3) if o.group_id not in initial else o
        for o in pool
    ]
    trace = adaptive_information_order(
        data.spec,
        data.metadata,
        data.fingerprint,
        TrainingPoolOracle(changed),
        groups,
        list(split["mandatory_hull_training_groups"]),
        0,
    )
    assert original["events"][0] == trace["events"][0]
    # Once the selected point is revealed, changed data may affect the next fitted state.
    assert original["events"][1]["parameters_before_selection"] != trace["events"][1]["parameters_before_selection"]


def test_all_orders_frozen_before_evaluation_and_fits_only_see_revealed_data(tmp_path, monkeypatch):
    import octwin.replay as module

    data, split, pool, _ = material()
    train_groups = set(split["train"])
    original_build = module.build_artifact
    fits = []

    def checked_build(view, observed):
        assert {o.group_id for o in view.observations} == {o.group_id for o in observed}
        assert {o.group_id for o in observed}.issubset(train_groups)
        fits.append(len({o.group_id for o in observed}))
        return original_build(view, observed)

    original_evaluate = module.evaluate_fixed_orders

    def checked_evaluate(*args):
        traces = json.loads((tmp_path / "selection_traces.json").read_text())["traces"]
        assert len(traces) == 3 and all(t["status"] == "completed" for t in traces)
        assert len(fits) == 3
        assert all(len(t["groups"]) == 12 for t in traces)
        return original_evaluate(*args)

    monkeypatch.setattr(module, "build_artifact", checked_build)
    monkeypatch.setattr(module, "evaluate_fixed_orders", checked_evaluate)
    result = run_adaptive_replay(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (9, 12), (0,))
    assert len(result["rows"]) == 18
    assert len(result["failures"]) == 3  # AHRI10 unavailable at9 groups, each strategy.
    assert result["selection_audit"]["adaptive_refits"] == 3
    assert result["selection_audit"]["max_revealed_groups"] == 12
    assert result["selection_audit"]["calibration_groups_used_by_selector"] == 0
    assert result["selection_audit"]["final_groups_used_by_selector"] == 0
    assert "Dotted: local D-optimal" in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_changed_final_answers_do_not_change_full_selection_traces(tmp_path):
    data, split, _, _ = material()
    run_adaptive_replay(DATA / "data.csv", DATA / "metadata.json", split, tmp_path / "original", (12,), (0,))
    changed_rows = [
        {
            **asdict(o.point),
            "group_id": o.group_id,
            "mass_flow_kg_s": o.mass_flow_kg_s * (2 if o.group_id in split["test"] else 1),
            "power_w": o.power_w * (2 if o.group_id in split["test"] else 1),
        }
        for o in data.observations
    ]
    write_csv(tmp_path / "changed.csv", changed_rows)
    write_json(tmp_path / "meta.json", data.metadata)
    run_adaptive_replay(tmp_path / "changed.csv", tmp_path / "meta.json", split, tmp_path / "changed", (12,), (0,))
    before = json.loads((tmp_path / "original/selection_traces.json").read_text())["traces"]
    after = json.loads((tmp_path / "changed/selection_traces.json").read_text())["traces"]
    assert before == after
    a = json.loads((tmp_path / "original/result.json").read_text())["rows"][0]
    b = json.loads((tmp_path / "changed/result.json").read_text())["rows"][0]
    assert a["mass_flow_kg_s_mare_percent"] != b["mass_flow_kg_s_mare_percent"]


def test_selection_failure_stops_without_fallback_or_extra_reveals(tmp_path, monkeypatch):
    import octwin.replay as module

    _, split, _, _ = material()

    def failure(*args):
        raise ValueError("forced local fit failure")

    monkeypatch.setattr(module, "build_artifact", failure)
    result = run_adaptive_replay(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (9, 12), (0,))
    traces = json.loads((tmp_path / "selection_traces.json").read_text())["traces"]
    trace = next(t for t in traces if t["strategy"] == "d_optimal")
    assert trace["status"] == "stopped_selector_failed"
    assert trace["reveal_log"] == split["mandatory_hull_training_groups"]
    assert trace["fit_attempts"] == 1
    rows = [r for r in result["rows"] if r["strategy"] == "d_optimal" and r["budget_groups"] == 12]
    assert len(rows) == 3 and all(r["status"] == "unavailable" for r in rows)
    assert result["selection_audit"]["selector_stops"] == 1


def test_initial_hull_must_cover_candidate_pool(tmp_path):
    _, split, _, _ = material()
    split = copy.deepcopy(split)
    split["mandatory_hull_training_groups"].pop()
    with pytest.raises(ValueError, match="do not cover"):
        run_adaptive_replay(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (9,), (0,))
    assert not (tmp_path / "protocol.json").exists()


def test_adaptive_protocol_is_not_silently_replaced(tmp_path):
    _, split, _, _ = material()
    run_adaptive_replay(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (9,), (0,))
    before = (tmp_path / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="protocol differs"):
        run_adaptive_replay(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (12,), (0,))
    assert (tmp_path / "protocol.json").read_bytes() == before


def test_adaptive_replay_cli_and_report(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "octwin",
            "adaptive-replay",
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
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert process.returncode == 0, process.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["kind"] == "adaptive_replay"
    assert result["selection_audit"]["total_orders"] == 3
    process = subprocess.run(
        [sys.executable, "-m", "octwin", "report", "--run", str(tmp_path), "--output", str(tmp_path / "rebuilt.html")],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert process.returncode == 0, process.stderr
    assert "LOCAL D-OPTIMAL ADAPTIVE REPLAY" in (tmp_path / "rebuilt.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["rolling_piston", "scroll"])
def test_two_structure_synthetic_adaptive_pipeline_is_not_real_validation(tmp_path, kind):
    from octwin.demo import generate

    rows, meta, split, _ = generate(tmp_path, kind)
    initial = ["g-0-0", "g-0-8", "g-8-0", "g-8-8"]
    other = [g for g in split["train"] if g not in initial][:5]
    small_split = {
        "train": initial + other,
        "calibration": split["calibration"][:1],
        "test": split["test"][:3],
        "mandatory_hull_training_groups": initial,
        "method": "L0 numerical software fixture, not real-machine validation",
    }
    allowed = set(small_split["train"] + small_split["calibration"] + small_split["test"])
    write_csv(tmp_path / "small.csv", [r for r in rows if r["group_id"] in allowed])
    result = run_adaptive_replay(
        tmp_path / "small.csv", tmp_path / "metadata.json", small_split, tmp_path / "replay", (4, 7, 9), (0,)
    )
    assert len(result["rows"]) == 27
    assert result["selection_audit"]["selector_stops"] == 0
    assert result["validation_role"] == "software_test"
    assert "SYNTHETIC SOFTWARE SELF-TEST ONLY" in (tmp_path / "replay/report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("fault", ["outside", "repeat", "omitted"])
def test_corrupted_selection_ledger_is_rejected_before_scoring(tmp_path, monkeypatch, fault):
    import octwin.replay as module

    _, split, _, _ = material()
    original = module.input_selection_order

    def corrupted(*args):
        order = original(*args)
        i = len(split["mandatory_hull_training_groups"])
        if fault == "outside":
            order[i] = split["test"][0]
            return order
        if fault == "repeat":
            order[i] = order[i - 1]
            return order
        return order[:i]

    def forbidden_scoring(*args):
        raise AssertionError("Final evaluator must not run for invalid selection ledger")

    monkeypatch.setattr(module, "input_selection_order", corrupted)
    monkeypatch.setattr(module, "evaluate_fixed_orders", forbidden_scoring)
    with pytest.raises(ValueError, match="Selection trace"):
        run_adaptive_replay(DATA / "data.csv", DATA / "metadata.json", split, tmp_path, (12,), (0,))
    assert not (tmp_path / "result.json").exists()


def test_invalid_acquisition_horizon_rejected_before_reveal():
    data, split, pool, groups = material()
    oracle = TrainingPoolOracle(pool)
    with pytest.raises(ValueError, match="Acquisition horizon"):
        adaptive_information_order(
            data.spec,
            data.metadata,
            data.fingerprint,
            oracle,
            groups,
            split["mandatory_hull_training_groups"],
            0,
            max_groups=3,
        )
    assert oracle.revealed_groups == []
