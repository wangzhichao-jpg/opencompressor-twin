"""Render report smoke-test with an existing Edge, all profile/cache/temp on D."""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
if not EDGE.is_file():
    raise SystemExit("Existing Microsoft Edge not found; inspect reports in a local browser")
TEMP = ROOT / ".tmp"
TEMP.mkdir(exist_ok=True)
env = {**os.environ, "TEMP": str(TEMP), "TMP": str(TEMP)}
parser = argparse.ArgumentParser()
parser.add_argument("--report", action="append", help="Project-relative report HTML; repeat for multiple reports")
args = parser.parse_args()
targets = (
    [(Path(p).parent.name, (ROOT / p).resolve()) for p in args.report]
    if args.report
    else [
        ("synthetic-scroll", ROOT / "runs/software_demo/scroll/run/report.html"),
        ("sanden-exploratory", ROOT / "runs/sanden_tgve08_dry/report.html"),
    ]
)
for name, path in targets:
    shot = TEMP / (name + ".png")
    command = [
        str(EDGE),
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
        "--user-data-dir=" + str(ROOT / ".cache/edge-report-check"),
        "--disk-cache-dir=" + str(ROOT / ".cache/edge-report-cache"),
        "--window-size=1440,2200",
        "--screenshot=" + str(shot),
        path.as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, timeout=40, env=env)
    if result.returncode or not shot.is_file():
        raise SystemExit(result.stderr.decode(errors="replace"))
    print(shot)
