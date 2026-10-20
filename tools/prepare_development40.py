"""Export ONLY the frozen original train40; no final/calibration labels in child.

This adapter never changes the original split. Source hashes go to an export
audit, not analytical metadata, so excluded-label changes cannot change the
child CSV/metadata/folds or downstream choices.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octwin.core import read_rows
from octwin.kernels import KERNEL_RIDGE, LENGTH_SCALE
from octwin.workflow import write_json


def export_development(data_path, meta_path, split_path, output):
    data_path, meta_path, split_path, output = map(Path, (data_path, meta_path, split_path, output))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = read_rows(data_path)
    all_groups = [g for key in ("train", "calibration", "test") for g in split[key]]
    if len(all_groups) != len(set(all_groups)) or set(all_groups) != {r["group_id"] for r in rows}:
        raise ValueError("Original split must uniquely partition every group")
    chosen = [r for r in rows if r["group_id"] in split["train"]]
    initial = split["mandatory_hull_training_groups"]
    if not set(initial).issubset(split["train"]) or len(set(initial)) != len(initial):
        raise ValueError("Invalid initial groups")
    remaining = sorted(set(split["train"]) - set(initial))
    np.random.default_rng(20261020).shuffle(remaining)
    folds = []
    for i, held in enumerate(np.array_split(remaining, 5)):
        validation = [str(g) for g in held]
        folds.append({"fold": i, "train": [g for g in split["train"] if g not in validation], "validation": validation})
    child = deepcopy(metadata)
    child.update(
        dataset_id=metadata["dataset_id"] + "-development-only",
        research_partition="development_only",
        parent_dataset_id=metadata["dataset_id"],
        admission_status="development_analysis_only_not_new_validation",
    )
    child["known_limitations"] = list(child.get("known_limitations", [])) + [
        "Contains only the original frozen training groups; CV is development research, not new blind validation"
    ]
    config = {
        "schema_version": 1,
        "scope": "development_only",
        "development_groups": split["train"],
        "forbidden_groups": split["calibration"] + split["test"],
        "initial_groups": initial,
        "fold_seed": 20261020,
        "folds": folds,
        "budgets": [9, 12, 16, 24],
        "seeds": [0, 1, 2, 3, 4],
        "strategies": ["random", "maximin", "d_optimal", "kernel_ivr"],
        "length_scale": LENGTH_SCALE,
        "kernel_ridge": KERNEL_RIDGE,
        "original_split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    # Frozen exports must not be silently replaced with different analytical data.
    for name, payload in (("metadata.json", child), ("development_protocol.json", config)):
        p = output / name
        if p.exists() and json.loads(p.read_text(encoding="utf-8")) != payload:
            raise ValueError("Development export differs; choose new directory")
    import csv
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(chosen[0]))
    writer.writeheader()
    writer.writerows(chosen)
    expected = buffer.getvalue()
    csv_path = output / "data.csv"
    if csv_path.exists() and csv_path.read_bytes() != expected.encode("utf-8"):
        raise ValueError("Development CSV differs; preserve frozen export")
    csv_path.write_bytes(expected.encode("utf-8"))
    write_json(output / "metadata.json", child)
    write_json(output / "development_protocol.json", config)
    write_json(
        output / "export_audit.json",
        {
            "source_csv_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "selected_group_ids": split["train"],
            "excluded_group_ids": config["forbidden_groups"],
            "selected_rows": len(chosen),
            "excluded_rows": len(rows) - len(chosen),
            "status": "development-only export; not an altered original train/test split",
        },
    )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public/arep21_r404a_65f/data.csv")
    parser.add_argument("--meta", default="data/public/arep21_r404a_65f/metadata.json")
    parser.add_argument("--split", default="data/public/arep21_r404a_65f/split.json")
    parser.add_argument("--output", default="data/development/arep21_train40")
    args = parser.parse_args()
    config = export_development(ROOT / args.data, ROOT / args.meta, ROOT / args.split, ROOT / args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "groups": len(config["development_groups"]),
                "fold_sizes": [len(f["validation"]) for f in config["folds"]],
            },
            indent=2,
        )
    )
