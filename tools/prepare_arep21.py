"""Prepare ORNL/AHRI AREP #21 Table A1, NOT Appendix B fitted map points.

Run from project root. Uses a manually/visually checked numerical transcription,
a frozen PDF checksum and input-only split. Never estimates shaft speed from Hz.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octwin.core import OperatingPoint
from octwin.properties import suction_properties
from octwin.reporting import write_csv
from octwin.workflow import write_json

PDF_SHA256 = "e2b1f34299e427a3927df6ddc6660253f1387e12e385c9fbab66a42a6743012f"
PDF = ROOT / "资料库/公开验证资料/第二轮筛选/AREP21.pdf"
TRANSCRIPTION = ROOT / "data_sources/arep21_table_a1.json"
OUT = ROOT / "data/public/arep21_r404a_65f"
PSI_PA = 6894.757293168
IN3_M3 = 0.000016387064
FT3_M3 = 0.028316846592
LB_KG = 0.45359237


def prepare():
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("Original AREP21 PDF fingerprint changed; recheck source before conversion")
    table = json.loads(TRANSCRIPTION.read_text(encoding="utf-8"))
    displacement = table["displacement_in3_rev"] * IN3_M3
    volume_rate = table["rated_displacement_ft3_h"] * FT3_M3 / 3600
    nominal_rpm = volume_rate / displacement * 60
    ts = (table["suction_temperature_f"] - 32) * 5 / 9 + 273.15
    rows, checks = [], []
    for block in table["blocks"]:
        n = len(block["power_w"])
        if not all(len(block[k]) == n for k in ("mass_flow_lb_h", "capacity_btu_h", "eer_btu_wh")):
            raise ValueError("Transcription columns have inconsistent lengths")
        first = block["first_suction_column"]
        if first < 0 or first + n > len(table["suction_pressure_psia"]):
            raise ValueError("Invalid source table column span")
        for i in range(n):
            col = first + i
            te = table["suction_dew_temperature_f"][col]
            td = block["discharge_dew_temperature_f"]
            identity = f"A1-Td{td}-Te{te}"
            eer_error = abs(block["capacity_btu_h"][i] / block["power_w"][i] - block["eer_btu_wh"][i])
            capacity, power, eer = block["capacity_btu_h"][i], block["power_w"][i], block["eer_btu_wh"][i]
            # C/P/E are independently rounded in print (1Btu/h,1W,0.01EER).
            if (capacity - 0.5) / (power + 0.5) > eer + 0.005 or (capacity + 0.5) / (power - 0.5) < eer - 0.005:
                raise ValueError(f"{identity}: capacity/power/EER inconsistent beyond printed rounding intervals")
            point = OperatingPoint(
                identity,
                table["suction_pressure_psia"][col] * PSI_PA,
                block["discharge_pressure_psia"] * PSI_PA,
                ts,
                nominal_rpm,
            )
            suction_properties(point, "R404A")
            rows.append(
                {
                    **asdict(point),
                    "group_id": identity,
                    "mass_flow_kg_s": block["mass_flow_lb_h"][i] * LB_KG / 3600,
                    "power_w": block["power_w"][i],
                    "source_pdf_page": 14,
                    "source_table": "A1",
                    "suction_dew_temperature_f": te,
                    "discharge_dew_temperature_f": td,
                    "source_capacity_btu_h": block["capacity_btu_h"][i],
                    "source_eer_btu_wh": block["eer_btu_wh"][i],
                }
            )
            checks.append(
                {
                    "point_id": identity,
                    "source_cell": f"PDF14/printed13 TableA1 Td={td}F Te={te}F, M/P rows",
                    "mass_flow_lb_h": block["mass_flow_lb_h"][i],
                    "power_w": block["power_w"][i],
                    "capacity_power_vs_eer_absolute_difference": eer_error,
                    "single_phase_gas_check": "passed",
                }
            )
    if len(rows) != 64:
        raise ValueError("Expected exactly64 populated Table A1 cells, not interpolated blank cells")
    # Freeze from input geometry only BEFORE fitting/scoring any model.
    coords = np.array([[r["suction_pressure_pa"], r["discharge_pressure_pa"]] for r in rows])
    normalized = (coords - coords.min(axis=0)) / np.ptp(coords, axis=0)
    hull_indices = set(map(int, ConvexHull(normalized).vertices))
    mandatory = [r["group_id"] for i, r in enumerate(rows) if i in hull_indices]
    rest = sorted(r["group_id"] for i, r in enumerate(rows) if i not in hull_indices)
    np.random.default_rng(42).shuffle(rest)
    split = {
        "train": mandatory + rest[24:],
        "calibration": rest[:12],
        "test": rest[12:24],
        "seed": 42,
        "method": "input-only convex-hull vertices observed + seeded interior group holdout; fixed65F/nominal-equivalent-speed interpolation ONLY",
        "mandatory_hull_training_groups": mandatory,
        "status": "retrospective published-experiment holdout, not prospective enterprise blind test",
    }
    meta = {
        "dataset_id": "arep21-copeland-zs21kae-r404a-65f-v1",
        "source_kind": "experiment",
        "validation_role": "public_validation",
        "admission_status": "reviewed_published_experiment_with_nominal_speed_limitation",
        "source": table["source_title"]
        + "; Appendix A Table A1, mass-flow/electrical-power test results; not Appendix B fitted coefficients",
        "source_url": table["source_url"],
        "source_location": table["location"],
        "accessed_at": "2026-10-19",
        "license": "Public ORNL/DOE technical report; no explicit dataset CC license stated. Numerical transcription with attribution; original report link-only.",
        "redistribution_status": "original_report_link_only; numerical_facts_attributed_no_CC_claim",
        "raw_sha256": PDF_SHA256,
        "transcription_sha256": hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest(),
        "transform_script": "tools/prepare_arep21.py",
        "transform_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pressure_basis": "absolute",
        "speed_definition": "nominal_mechanical_rpm",
        "nominal_speed_provenance": {
            "source_location": "AREP21 PDF6/printed5 sec2b, 3.11in3/rev and377.71ft3/hr",
            "volume_rate_m3_s": volume_rate,
            "definition": "60 * rated_volume_rate_m3_s / displacement_m3_rev; NOT frequency conversion, NOT measured per-point speed",
        },
        "power_boundary": "electrical_input",
        "suction_regime": "fixed_actual_suction_temperature_65F",
        "supply_frequency_hz": 60,
        "compressor": {
            "compressor_type": "scroll",
            "manufacturer": "Copeland",
            "model_id": "ZS21KAE-PFV",
            "refrigerant": "R404A",
            "displacement_m3_rev": displacement,
        },
        "measurement_boundary": "Refrigerant flow via Micro Motion Elite; compressor electrical power via Yokogawa PR300;230V; chamber95+/-1F;original POE lubricant",
        "uncertainty_source": "Report Table2 PDF8: flow +/-0.10%, power +/-0.5%,RTD +/-0.4F, pressure0.25%FS; accuracy specifications, NOT converted to standard uncertainty",
        "known_limitations": [
            "Speed is nominal-equivalent from published rated displacement rate, not per-point shaft measurement; no cross-speed or absolute volumetric-efficiency validation",
            "Published header pressures rounded to whole psia; used as-is, not silently replaced with simulated pressures",
            "Source used REFPROP9.0, current model CoolProp8.0.0; property/pressure-rounding differences remain",
            "Only R404A TableA1 fixed65F slice; no pooling of alternate fluids, other superheat regimes or manufacturer-map points",
            "Source capacities normalized to zero subcooling; retained for extraction audit only, not used as independent supervision or claimed predicted cooling capacity",
            "Original hermetic compressor contains POE oil; oil circulation/correction not reported; no oil-mixture model or oil-free claim",
            "Input hull vertices intentionally observed; evaluation tests interior interpolation, not boundary or cross-speed generalization",
            "Single public report/single machine; held-out conditions share laboratory/instrument calibration and do not prove cross-brand transfer",
        ],
    }
    if (OUT / "split.json").exists() and json.loads((OUT / "split.json").read_text(encoding="utf-8")) != split:
        raise ValueError("Existing frozen split differs; use a new dataset/protocol ID, do not overwrite")
    write_csv(OUT / "data.csv", rows)
    write_json(OUT / "metadata.json", meta)
    write_json(OUT / "split.json", split)
    write_json(
        OUT / "extraction_audit.json",
        {
            "raw_sha256": PDF_SHA256,
            "rows": checks,
            "count": len(rows),
            "nominal_rpm": nominal_rpm,
            "split_created_before_scoring": True,
            "blank_cells": "excluded, not filled",
            "visual_check": "Original PDF6 specs and PDF14 entire A1 table viewed; numbers transcribed, units checked; no OCR",
        },
    )
    write_json(ROOT / "data_manifests/arep21_r404a_65f.json", meta)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "nominal_equivalent_rpm": nominal_rpm,
                "group_counts": {k: len(split[k]) for k in ("train", "calibration", "test")},
                "mandatory_input_hull_groups": mandatory,
                "output": str(OUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare()
