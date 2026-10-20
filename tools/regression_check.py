"""Numerical regression gate: recompute frozen results into .tmp and require exact equality.

Re-scoring already-seen data here is a software regression check, NOT a new
evaluation. Used before/after code hygiene changes (formatting, lint fixes).
"""

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octwin.core import load_dataset  # noqa: E402
from octwin.development import run_development_study  # noqa: E402
from octwin.discrepancy import add_residual_correction  # noqa: E402
from octwin.workflow import benchmark, build_artifact, calibrate_interval, metrics, predict_point  # noqa: E402

TMP = ROOT / ".tmp/regression"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def same(label, new, old, failures):
    if json.dumps(new, sort_keys=True) != json.dumps(old, sort_keys=True):
        failures.append(label)
    print(("OK   " if label not in failures else "FAIL ") + label)


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    failures = []
    base = "data/public/arep21_r404a_65f"
    run = benchmark(ROOT / base / "data.csv", ROOT / base / "metadata.json", TMP / "arep21", load(base + "/split.json"))
    old = load("evidence/arep21_r404a_65f/result.json")
    same("AREP21 four-parameter metrics", run["metrics"], old["metrics"], failures)
    same("AREP21 affine metrics", run["baseline"]["metrics"], old["baseline"]["metrics"], failures)
    same("AREP21 AHRI10 metrics", run["ahri10_baseline"]["metrics"], old["ahri10_baseline"]["metrics"], failures)

    frozen = load("evidence/round5_external/result.json")["datasets"]
    for name in ("arep11_r410a_65f", "arep11_r32_65f"):
        folder = ROOT / "data/public" / name
        split = load(f"data/public/{name}/split.json")
        run = benchmark(folder / "data.csv", folder / "metadata.json", TMP / name, split)
        data = load_dataset(folder / "data.csv", folder / "metadata.json")
        train = [o for o in data.observations if o.group_id in split["train"]]
        calibration = [o for o in data.observations if o.group_id in split["calibration"]]
        test = [o for o in data.observations if o.group_id in split["test"]]
        corrected = add_residual_correction(build_artifact(replace(data, observations=train), train), train)
        corrected["interval"] = calibrate_interval(corrected, calibration)
        rbf = metrics(test, [predict_point(corrected, o.point) for o in test])
        methods = frozen[name]["methods"]
        same(f"{name} four-parameter", run["metrics"], methods["semiempirical"], failures)
        same(f"{name} RBF extension", rbf, methods["semiempirical_rbf"], failures)
        same(f"{name} affine", run["baseline"]["metrics"], methods["affine"], failures)
        same(f"{name} AHRI10", run["ahri10_baseline"]["metrics"], methods["ahri10"], failures)

    dev = "data/development/arep21_train40"
    study = run_development_study(
        ROOT / dev / "data.csv",
        ROOT / dev / "metadata.json",
        load(dev + "/development_protocol.json"),
        TMP / "development",
    )
    old = load("evidence/development/result.json")
    same("development OOF metrics", study["oof_metrics"], old["oof_metrics"], failures)
    same("development design summary", study["summary"], old["summary"], failures)
    same("development trace hash", study["trace_canonical_sha256"], old["trace_canonical_sha256"], failures)
    if failures:
        raise SystemExit(f"{len(failures)} regression failure(s): {failures}")
    print("All frozen numerical results reproduced exactly")


if __name__ == "__main__":
    main()
