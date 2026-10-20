"""Prepare a conservative, exploratory dry subset; never fabricate electrical power.

Run from project root:
  .venv/Scripts/python.exe tools/prepare_sanden.py
Original CSV: Leclercq and Applied Thermodynamics Laboratory, University of Liege,
Zenodo record 14603617, CC BY 4.0; related DOI 10.1016/j.ijrefrig.2025.05.013.
This subset is NOT a prospectively independent laboratory blind test. The paper
uses dry points for instrumentation calibration. Group IDs are conservative
input-only near-repeat clusters, not recovered original nominal test IDs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from octwin.core import OperatingPoint
from octwin.properties import suction_properties
from octwin.reporting import write_csv
from octwin.workflow import write_json

RAW = ROOT / "资料库/公开验证资料/启动筛选/sanden_no_oil.csv"
OUT = ROOT / "data/public/sanden_tgve08_dry"
MD5 = "5325b49d098b133ce00d878080d00891"


def prepare():
    raw_bytes = RAW.read_bytes()
    if hashlib.md5(raw_bytes).hexdigest() != MD5:
        raise ValueError("Raw dataset MD5 differs from Zenodo metadata; stop for source review")
    with RAW.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter=";"))
    kept, audit = [], []
    for line, source in enumerate(raw, 2):
        r = {k: float(v) for k, v in source.items()}
        # Predeclared source/phase filters only, independent of model error or targets.
        # A stricter subset than the paper's 23 superheated points: no liquid feed,
        # no oil, quality=1 and margin away from saturation.
        reasons = []
        if r["x_oil_value"] != 0 or r["m_dot_oil"] != 0:
            reasons.append("nonzero oil")
        if r["Q_inlet_value"] < 0.999999:
            reasons.append("inlet quality below one")
        if r["SH_inlet_value"] <= max(5.0, 3 * r["SH_inlet_uncertainty"]):
            reasons.append("insufficient superheat margin")
        if abs(r["m_dot_liq_value"]) >= 0.1:
            reasons.append("liquid feed >=0.1 g/s; excluded conservatively")
        point = OperatingPoint(
            f"zenodo14603617-line{line}",
            r["P_su_cp_value"] * 1e5,
            r["P_ex_cp_value"] * 1e5,
            r["T_su_cp_value"] + 273.15,
            r["speed_comp_value"],
        )
        if not reasons:
            try:
                suction_properties(point, "R1233zd(E)")
            except ValueError as exc:
                reasons.append(str(exc))
        audit.append({"source_csv_line": line, "included": not reasons, "reasons": reasons})
        if reasons:
            continue
        # Compressor speed, not motor speed (pulley ratio 1.7, paper section 2.1).
        # W_dot_shaft is compressor shaft power after belt loss, paper Eq.12.
        from dataclasses import asdict

        kept.append(
            {
                **asdict(point),
                "source_csv_line": line,
                "mass_flow_kg_s": r["m_dot_tot_value"] * 0.001,
                "power_w": r["W_dot_shaft_value"],
                "mass_flow_uncertainty_kg_s": r["m_dot_tot_uncertainty"] * 0.001,
                "power_uncertainty_w": r["W_dot_shaft_uncertainty"],
            }
        )
    # Connected-component grouping: near nominal pressures/speed stay together.
    # Thresholds are declared engineering grouping choices, not a GUM calculation.
    parent = list(range(len(kept)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(kept):
        for j, b in enumerate(kept[:i]):
            if (
                abs(a["suction_pressure_pa"] - b["suction_pressure_pa"]) <= 10000
                and abs(a["discharge_pressure_pa"] - b["discharge_pressure_pa"]) <= 20000
                and abs(a["speed_rpm"] - b["speed_rpm"]) <= 50
            ):
                parent[find(i)] = find(j)
    for i, r in enumerate(kept):
        r["group_id"] = f"near-repeat-line{kept[find(i)]['source_csv_line']}"
    groups = sorted({r["group_id"] for r in kept})
    # Freeze two final groups by hash, never by observed response/model error.
    ordered = sorted(groups, key=lambda g: hashlib.sha256(("octwin-sanden-v1-seed42:" + g).encode()).hexdigest())
    split = {
        "train": ordered[2:],
        "calibration": [],
        "test": ordered[:2],
        "seed": 42,
        "method": "input-only near-repeat groups; SHA256 order; two final groups; no residual calibration set",
        "status": "exploratory frozen split; not a prospective blind test",
    }
    sha = hashlib.sha256(raw_bytes).hexdigest()
    meta = {
        "dataset_id": "sanden-tgve08-dry-shaft-exploratory-v1",
        "source_kind": "experiment",
        "validation_role": "public_validation",
        "admission_status": "exploratory_only",
        "source": "Leclercq, Bederna and Lemort; Zenodo record 14603617 sanden_no_oil.csv; related paper DOI 10.1016/j.ijrefrig.2025.05.013",
        "source_url": "https://zenodo.org/records/14603617",
        "license": "CC BY 4.0; attribution to Applied Thermodynamics Laboratory, University of Liege and Nicolas Leclercq",
        "pressure_basis": "absolute",
        "speed_definition": "mechanical_rpm",
        "power_boundary": "compressor_shaft_after_belt_loss_NOT_electrical_input",
        "compressor": {
            "compressor_type": "scroll",
            "manufacturer": "Sanden",
            "model_id": "TGVE08-retrofitted-open-drive",
            "refrigerant": "R1233zd(E)",
            "displacement_m3_rev": 86e-6,
        },
        "raw_sha256": sha,
        "raw_md5": MD5,
        "source_version": "Zenodo V1, modified 2026-05-29",
        "accessed_at": "2026-10-19",
        "redistribution_status": "allowed_with_attribution",
        "reference_enthalpy_difference": "not provided; no capacity/COP comparison",
        "transform_script": "tools/prepare_sanden.py",
        "source_location": "CSV source_csv_line; paper sec2.1/2.2, Table1, sec3.1, Eq9/11/12",
        "grouping": "Connected input clusters: |dps|<=0.1bar, |dpd|<=0.2bar, |dn|<=50rpm; ignore temperature conservatively; original nominal IDs unavailable",
        "uncertainty_source": "Adjacent source columns retained, not used as fitting weights; no metrological interpretation assumed",
        "known_limitations": [
            "Dry points used for instrumentation calibration by source authors; common calibration/systematic dependence remains",
            "Retrofitted open-drive machine, no electrical input measurement; cannot validate hermetic electrical losses",
            "Conservative subset excludes all oil and liquid-fed cases; not the whole two-phase campaign",
            "Source PDF fetched directly failed; source sections read through Tavily text extraction, no original-page visual crosscheck",
            "Few groups, two speed clusters and sparse 4D input coverage; hull refusal may preclude a meaningful holdout score",
            "This preparation and split were fixed before model scoring; no outcome-driven retuning",
        ],
    }
    write_csv(OUT / "data.csv", kept)
    write_json(OUT / "metadata.json", meta)
    write_json(OUT / "split.json", split)
    write_json(
        OUT / "extraction_audit.json",
        {"raw_rows": len(raw), "included_rows": len(kept), "groups": len(groups), "raw_sha256": sha, "rows": audit},
    )
    write_json(ROOT / "data_manifests/sanden_tgve08_dry.json", meta)
    print(
        json.dumps(
            {
                "raw_rows": len(raw),
                "selected_rows": len(kept),
                "groups": len(groups),
                "train_groups": len(split["train"]),
                "test_groups": split["test"],
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare()
