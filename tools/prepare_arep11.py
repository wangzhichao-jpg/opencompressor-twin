"""Prepare AREP #11 Tables A3 (R410A) and A6 (R32) under the frozen round-5 protocol.

Input-only split rule from data_sources/round5_external_protocol.json. Does not
fit or score anything. Refuses to overwrite a different frozen split.
"""

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octwin.core import OperatingPoint  # noqa: E402
from octwin.properties import suction_properties  # noqa: E402
from octwin.reporting import write_csv  # noqa: E402
from octwin.workflow import write_json  # noqa: E402

PSI_PA = 6894.757293168
IN3_M3 = 0.000016387064
FT3_M3 = 0.028316846592
LB_KG = 0.45359237
PDF_SHA256 = "7da8c43e3c3f8ba3905a016f56a6205db2049ed7c5e7403db70fbe4ddbafc291"
DATASETS = {
    "arep11_r410a_65f": ("arep11_r410a_table_a3.json", "R410A"),
    "arep11_r32_65f": ("arep11_r32_table_a6.json", "R32"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(name, source_file, refrigerant):
    table_path = ROOT / "data_sources" / source_file
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if table["pdf_sha256"] != PDF_SHA256 or table["refrigerant"] != refrigerant:
        raise ValueError("Unexpected transcription source")
    displacement = table["displacement_in3_rev"] * IN3_M3
    volume_rate = table["rated_displacement_ft3_h"] * FT3_M3 / 3600
    nominal_rpm = volume_rate / displacement * 60
    ts = (table["suction_temperature_f"] - 32) * 5 / 9 + 273.15
    rows, checks = [], []
    for block in table["blocks"]:
        first, td = block["first_suction_column"], block["discharge_dew_temperature_f"]
        for i, power in enumerate(block["power_w"]):
            col = first + i
            te = table["suction_dew_temperature_f"][col]
            identity = f"{table['table']}-Td{td}-Te{te}"
            capacity, eer = block["capacity_btu_h"][i], block["eer_btu_wh"][i]
            # C (1 Btu/h), P (1 W), E (0.01) are independently rounded in print.
            if (capacity - 0.5) / (power + 0.5) > eer + 0.005 or (capacity + 0.5) / (power - 0.5) < eer - 0.005:
                raise ValueError(f"{identity}: C/P/EER inconsistent beyond printed rounding")
            point = OperatingPoint(
                identity,
                table["suction_pressure_psia"][col] * PSI_PA,
                block["discharge_pressure_psia"] * PSI_PA,
                ts,
                nominal_rpm,
            )
            suction_properties(point, refrigerant)
            rows.append(
                {
                    **asdict(point),
                    "group_id": identity,
                    "mass_flow_kg_s": block["mass_flow_lb_h"][i] * LB_KG / 3600,
                    "power_w": float(power),
                    "suction_dew_temperature_f": te,
                    "discharge_dew_temperature_f": td,
                }
            )
            checks.append(
                {
                    "point_id": identity,
                    "mass_flow_lb_h": block["mass_flow_lb_h"][i],
                    "power_w": power,
                    "cpe_rounding_check": "passed",
                    "single_phase_gas_check": "passed",
                }
            )
    # Frozen protocol split: input geometry only.
    coords = np.array([[r["suction_pressure_pa"], r["discharge_pressure_pa"]] for r in rows])
    normalized = (coords - coords.min(axis=0)) / np.ptp(coords, axis=0)
    hull = set(map(int, ConvexHull(normalized).vertices))
    mandatory = [r["group_id"] for i, r in enumerate(rows) if i in hull]
    rest = sorted(r["group_id"] for i, r in enumerate(rows) if i not in hull)
    n = max(9, round(0.1875 * len(rows)))
    if len(rows) < len(mandatory) + 2 * 9 + 4:
        raise ValueError("Too few groups for the frozen test protocol; description only")
    np.random.default_rng(20261021).shuffle(rest)
    split = {
        "train": mandatory + rest[2 * n :],
        "calibration": rest[n : 2 * n],
        "test": rest[:n],
        "seed": 20261021,
        "mandatory_hull_training_groups": mandatory,
        "method": "round5 frozen protocol: hull vertices train; seeded remainder test/calibration/train",
        "protocol_id": "round5-external-frozen-v1",
        "status": "new external holdout; split frozen before any model fit or score on this dataset",
    }
    meta = {
        "dataset_id": f"{name}-v1",
        "source_kind": "experiment",
        "validation_role": "public_validation",
        "admission_status": "reviewed_published_experiment_with_nominal_speed_limitation",
        "source": table["source_title"]
        + f"; Appendix A Table {table['table']} measured results; not Appendix B coefficients",
        "source_url": table["source_url"],
        "source_location": table["location"],
        "accessed_at": "2026-10-20",
        "license": "Public ORNL/DOE technical report; no explicit dataset CC license stated. Numerical transcription with attribution; original report link-only.",
        "redistribution_status": "pending release decision; numerical facts attributed, no CC claim",
        "raw_sha256": PDF_SHA256,
        "transcription_sha256": sha(table_path),
        "transform_script": "tools/prepare_arep11.py",
        "transform_script_sha256": sha(__file__),
        "pressure_basis": "absolute",
        "speed_definition": "nominal_mechanical_rpm",
        "nominal_speed_provenance": {
            "source_location": "AREP11 PDF6/printed5 sec2b, 1.24 in3/rev and 150.69 ft3/hr",
            "volume_rate_m3_s": volume_rate,
            "definition": "60 * rated_volume_rate_m3_s / displacement_m3_rev; NOT frequency conversion, NOT measured per-point speed",
        },
        "power_boundary": "electrical_input",
        "suction_regime": "fixed_actual_suction_temperature_65F",
        "supply_frequency_hz": 60,
        "compressor": {
            "compressor_type": "scroll",
            "manufacturer": "Copeland",
            "model_id": "ZP21K5E-PFV",
            "refrigerant": refrigerant,
            "displacement_m3_rev": displacement,
        },
        "measurement_boundary": "Micro Motion Elite mass flow; Yokogawa PR300 compressor power; chamber 95+/-1F; original POE",
        "known_limitations": [
            "Same ORNL laboratory/instrument family as AREP21; different compressor model, not an independent laboratory",
            "Speed is nominal-equivalent from rated displacement rate, not per-point measured speed",
            "Header pressures printed to whole psia; used as printed",
            "Source used REFPROP; this program uses CoolProp (R410A as pseudo-pure)",
            "Fixed 65F suction temperature only; superheat tables and DR-5/L-41A excluded by frozen protocol",
            "Hull vertices observed in training; holdout tests interior interpolation only",
        ],
    }
    out = ROOT / "data/public" / name
    if (out / "split.json").exists() and json.loads((out / "split.json").read_text(encoding="utf-8")) != split:
        raise ValueError("Existing frozen split differs; do not overwrite")
    write_csv(out / "data.csv", rows)
    write_json(out / "metadata.json", meta)
    write_json(out / "split.json", split)
    write_json(
        out / "extraction_audit.json",
        {
            "raw_sha256": PDF_SHA256,
            "rows": checks,
            "count": len(rows),
            "nominal_rpm": nominal_rpm,
            "blank_cells": "excluded, not filled",
            "visual_check": "Rendered pages 15 and 18 compared block-by-block with native text-layer transcription",
        },
    )
    write_json(ROOT / "data_manifests" / f"{name}.json", meta)
    return {
        "rows": len(rows),
        "hull": len(mandatory),
        "n_test": n,
        "train": len(split["train"]),
        "nominal_rpm": nominal_rpm,
    }


if __name__ == "__main__":
    print(json.dumps({k: prepare(k, *v) for k, v in DATASETS.items()}, indent=2))
