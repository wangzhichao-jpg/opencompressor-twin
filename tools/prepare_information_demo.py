"""Prepare an offline initial-hull model and INPUT-ONLY approved-pool demo.

No new physical experiments/validation claim. Uses an existing frozen training
pool; final/calibration observations never enter the fit or candidate CSV.
"""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octwin.core import load_dataset
from octwin.reporting import write_csv
from octwin.workflow import build_artifact, check_split, recommend, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/development/arep21_train40/data.csv")
parser.add_argument("--meta", default="data/development/arep21_train40/metadata.json")
parser.add_argument("--split", default="data/development/arep21_train40/development_protocol.json")
parser.add_argument("--output", default="runs/information_demo_dev3")
args = parser.parse_args()
data = load_dataset(ROOT / args.data, ROOT / args.meta)
split = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
if split.get("scope") == "development_only":
    from octwin.development import validate_development

    validate_development(data, split)
    initial, train_groups = split["initial_groups"], split["development_groups"]
    exclusions = {"excluded_original_groups": split["forbidden_groups"]}
else:
    check_split(data.observations, split)
    initial, train_groups = split.get("mandatory_hull_training_groups", []), split["train"]
    exclusions = {"excluded_calibration_groups": split["calibration"], "excluded_final_groups": split["test"]}
if len(initial) < 4 or len(initial) != len(set(initial)) or not set(initial).issubset(train_groups):
    raise ValueError("Frozen split needs >=4 unique initial hull groups inside train pool")
observed = [o for o in data.observations if o.group_id in initial]
candidates = [o.point for o in data.observations if o.group_id in train_groups and o.group_id not in initial]
artifact = build_artifact(replace(data, observations=observed), observed)
output = ROOT / args.output
if (output / "model_initial.json").exists() and json.loads(
    (output / "model_initial.json").read_text(encoding="utf-8")
) != artifact:
    raise ValueError("Existing demo model differs; use a new output directory, preserve old evidence")
write_json(output / "model_initial.json", artifact)
write_csv(output / "candidate_inputs.csv", [{**asdict(p), "relative_cost": 1} for p in candidates])
result = recommend(artifact, candidates, 1, "d_optimal", {p.point_id: 1 for p in candidates})
write_json(output / "recommendation.json", result)
write_json(
    output / "kernel_recommendation.json",
    recommend(artifact, candidates, 1, "kernel_ivr", {p.point_id: 1 for p in candidates}),
)
write_json(
    output / "demo_scope.json",
    {
        "type": "offline recommendation demonstration using existing published training pool; NOT new validation or bench-time result",
        "initial_groups": initial,
        "candidate_point_ids": [p.point_id for p in candidates],
        **exclusions,
        "dataset_fingerprint": data.fingerprint,
        "cost_semantics": "unit relative cost, not measured test time",
    },
)
print(
    json.dumps(
        {
            "output": str(output),
            "initial_groups": len(initial),
            "candidates": len(candidates),
            "selected": result["selected"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
