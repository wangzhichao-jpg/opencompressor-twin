"""Transcribe AREP #11 Tables A3/A6 from the PDF native text layer by cell position.

Output is checked against rendered pages (visual review) and printed-rounding
C/P/E consistency. Blank cells are excluded. No OCR, no Appendix B coefficients.
"""

import hashlib
import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "资料库/公开验证资料/第五轮筛选/AREP11.pdf"
PDF_SHA256 = "7da8c43e3c3f8ba3905a016f56a6205db2049ed7c5e7403db70fbe4ddbafc291"
TABLES = {
    "A3": {"page": 15, "printed": 14, "refrigerant": "R410A", "out": "arep11_r410a_table_a3.json"},
    "A6": {"page": 18, "printed": 17, "refrigerant": "R32", "out": "arep11_r32_table_a6.json"},
}
ROWS = {"C": "capacity_btu_h", "P": "power_w", "M": "mass_flow_lb_h", "E": "eer_btu_wh"}


def number(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse(page):
    words = page.get_text("words")
    # Header: "10 (76)" pairs on one line; collect "(nnn)" tokens with x positions.
    header_y = min(w[1] for w in words if w[4].startswith("(") and w[4][1:-1].isdigit())
    header = sorted((w for w in words if abs(w[1] - header_y) < 2), key=lambda w: w[0])
    columns = []
    for i, w in enumerate(header):
        if w[4].startswith("(") and w[4].endswith(")"):
            dew = int(header[i - 1][4])
            columns.append({"dew_f": dew, "psia": int(w[4][1:-1]), "x": (header[i - 1][0] + w[2]) / 2})
    if len(columns) != 10:
        raise ValueError("Expected 10 suction columns")
    # Row labels C/P/A/M/E/%/T and block labels "140" + "(556)".
    labels = sorted(
        (w for w in words if w[4] in {"C", "P", "A", "M", "E", "%", "T"} and w[0] < columns[0]["x"] - 20),
        key=lambda w: w[1],
    )
    blocks_raw = sorted(
        (
            w
            for w in words
            if w[0] < labels[0][0] - 5 and (w[4].isdigit() or (w[4].startswith("(") and w[4][1:-1].isdigit()))
        ),
        key=lambda w: w[1],
    )
    block_heads = []
    for w in blocks_raw:
        if w[4].isdigit() and int(w[4]) in range(60, 150):
            press = min(
                (v for v in blocks_raw if v[4].startswith("(") and 0 < v[1] - w[1] < 20), key=lambda v: v[1] - w[1]
            )
            block_heads.append({"td": int(w[4]), "psia": int(press[4][1:-1]), "y": (w[1] + press[3]) / 2})
    if len(block_heads) != 8:
        raise ValueError(f"Expected 8 discharge blocks, got {len(block_heads)}")
    numbers = [w for w in words if number(w[4]) is not None and w[0] > columns[0]["x"] - 40]
    result = []
    for b in block_heads:
        rows = [lab for lab in labels if min(block_heads, key=lambda h: abs((lab[1] + lab[3]) / 2 - h["y"])) is b]
        if len(rows) != 7:
            raise ValueError(f"Block {b['td']}: expected 7 row labels")
        cells = {}
        for lab in rows:
            y = (lab[1] + lab[3]) / 2
            for w in numbers:
                if abs((w[1] + w[3]) / 2 - y) < 3:
                    col = min(range(10), key=lambda c: abs((w[0] + w[2]) / 2 - columns[c]["x"]))
                    if abs((w[0] + w[2]) / 2 - columns[col]["x"]) > 40:
                        raise ValueError("Unassigned numeric cell")
                    key = (lab[4], col)
                    if key in cells:
                        raise ValueError("Duplicate numeric cell")
                    cells[key] = number(w[4])
        used = sorted({c for (_, c) in cells})
        if used != list(range(used[0], used[-1] + 1)):
            raise ValueError("Non-contiguous populated columns")
        block = {
            "discharge_dew_temperature_f": b["td"],
            "discharge_pressure_psia": b["psia"],
            "first_suction_column": used[0],
        }
        for short, name in ROWS.items():
            values = [cells.get((short, c)) for c in used]
            if any(v is None for v in values):
                raise ValueError(f"Block {b['td']}: missing {short} value in populated column")
            block[name] = [int(v) if name in {"capacity_btu_h", "power_w"} else v for v in values]
        result.append(block)
    return columns, sorted(result, key=lambda r: -r["discharge_dew_temperature_f"])


def main():
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != PDF_SHA256:
        raise SystemExit("AREP11 PDF fingerprint changed")
    doc = fitz.open(PDF)
    for table, info in TABLES.items():
        columns, blocks = parse(doc[info["page"] - 1])
        payload = {
            "source_url": "https://info.ornl.gov/sites/publications/files/Pub44928.pdf",
            "source_title": "AHRI Low-GWP AREP Test Report #11, Compressor Calorimeter Test of R-410A Alternatives R-32, DR-5, and L-41A Using a Scroll Compressor, ORNL",
            "location": f"PDF physical page {info['page']}, printed page {info['printed']}, Table {table}; specification on PDF physical page 6, printed page 5",
            "table": table,
            "refrigerant": info["refrigerant"],
            "source_kind": "experiment",
            "note": "Native PDF text-layer transcription by cell position, checked against rendered page; NOT Appendix B polynomial points; no CC dataset license asserted",
            "pdf_sha256": PDF_SHA256,
            "suction_dew_temperature_f": [c["dew_f"] for c in columns],
            "suction_pressure_psia": [c["psia"] for c in columns],
            "suction_temperature_f": 65,
            "displacement_in3_rev": 1.24,
            "rated_displacement_ft3_h": 150.69,
            "compressor_model": "Copeland ZP21K5E-PFV",
            "blocks": blocks,
        }
        out = ROOT / "data_sources" / info["out"]
        if out.exists() and json.loads(out.read_text(encoding="utf-8")) != payload:
            raise SystemExit(f"{out.name} differs from frozen transcription; do not overwrite")
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(table, info["refrigerant"], "cells", sum(len(b["power_w"]) for b in blocks))
        for b in blocks:
            print(
                " Td",
                b["discharge_dew_temperature_f"],
                f"({b['discharge_pressure_psia']})",
                "first col",
                b["first_suction_column"],
                "P",
                b["power_w"],
                "M",
                b["mass_flow_lb_h"],
            )


if __name__ == "__main__":
    main()
