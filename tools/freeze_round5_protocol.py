"""Record protocol/code hashes BEFORE downloading round-5 candidate data."""

import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_sources/round5_protocol_freeze.json"
PDF = ROOT / "资料库/公开验证资料/第五轮筛选/AREP11.pdf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if OUT.exists():
    raise SystemExit("Protocol freeze record already exists; do not overwrite")
if PDF.exists():
    raise SystemExit("Candidate PDF already present; freeze must precede download")
files = ["docs/第五轮外部检验协议.md", "data_sources/round5_external_protocol.json"]
record = {
    "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "candidate_pdf_downloaded": False,
    "protocol_files": {f: digest(ROOT / f) for f in files},
    "evaluation_code_sha256": {
        p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT / "src/octwin").glob("*.py"))
    },
}
OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record["protocol_files"], ensure_ascii=False, indent=1), record["frozen_at_utc"])
