"""Reproducible SYNTHETIC software demonstration, NOT compressor validation."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import CompressorSpec, OperatingPoint
from .models import get_model
from .reporting import write_csv
from .workflow import benchmark, load_artifact, recommend, write_json


def generate(output, kind, seed=42):
    output = Path(output)
    spec = CompressorSpec(
        kind, "SYNTHETIC-NOT-A-BRAND", "software-grid-" + kind, "R134a", 9e-6 if kind == "rolling_piston" else 25e-6
    )
    params = dict(v_intercept=2.0, v_loss=0.12, is_intercept=1.1, is_curvature=0.35)
    rng = np.random.default_rng(seed)
    rows, boundary, interior = [], [], []
    for i, ps in enumerate(np.linspace(250000, 450000, 9)):
        for j, pd in enumerate(np.linspace(1100000, 1800000, 9)):
            group = f"g-{i}-{j}"
            point = OperatingPoint(group, float(ps), float(pd), 300.0, 3600.0)
            pred = get_model(kind).predict(spec, point, params)
            row = {
                **asdict(point),
                "group_id": group,
                "mass_flow_kg_s": pred["mass_flow_kg_s"] * float(1 + rng.normal(0, 0.005)),
                "power_w": pred["power_w"] * float(1 + rng.normal(0, 0.005)),
            }
            rows.append(row)
            (boundary if i in (0, 8) or j in (0, 8) else interior).append(group)
    rng.shuffle(interior)
    split = {
        "train": boundary + interior[27:],
        "calibration": interior[:12],
        "test": interior[12:27],
        "seed": seed,
        "method": "synthetic demonstration: observed boundary, disjoint interior group holdout; not boundary generalization",
    }
    meta = {
        "dataset_id": "synthetic-" + kind,
        "source_kind": "synthetic",
        "validation_role": "software_test",
        "source": "octwin.demo.generate: same-family equations plus independent 0.5% Gaussian noise; software self-test only",
        "license": "Generated software-test fixture; not manufacturer data",
        "pressure_basis": "absolute",
        "speed_definition": "mechanical_rpm",
        "power_boundary": "synthetic electrical input",
        "compressor": asdict(spec),
        "known_limitations": "Same-family synthetic generator cannot prove real model adequacy",
    }
    write_csv(output / "data.csv", rows)
    write_json(output / "metadata.json", meta)
    write_json(output / "split.json", split)
    candidates = [
        asdict(OperatingPoint(f"candidate-{k}", 270000.0 + k * 33000.0, 1200000.0 + k * 75000.0, 300.0, 3600.0))
        for k in range(5)
    ]
    write_csv(output / "candidates.csv", candidates)
    return rows, meta, split, candidates


def run_demo(output):
    summaries = []
    for kind in ("rolling_piston", "scroll"):
        folder = Path(output) / kind
        _, _, split, raw_candidates = generate(folder, kind)
        result = benchmark(folder / "data.csv", folder / "metadata.json", folder / "run", split)
        model = load_artifact(folder / "run" / "model.json")
        selected = recommend(model, [OperatingPoint(**p) for p in raw_candidates], 1)
        write_json(folder / "run" / "recommendation.json", selected)
        summaries.append(
            {
                "compressor_type": kind,
                "evidence": result["evidence_label"],
                "metrics": result["metrics"],
                "report": str(folder / "run" / "report.html"),
            }
        )
    write_json(Path(output) / "summary.json", summaries)
    return summaries
