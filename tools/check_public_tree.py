"""Check what Git would publish from a work tree.

Generic hard rules: local environments/outputs, original literature files,
oversized files, personal absolute paths, credential-like assignments, missing
LICENSE. Project-specific confidentiality rules (forbidden path prefixes and
terms) are read from an optional, Git-ignored JSON file passed with --rules, so
that the public code itself does not need to name what must stay private.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    "private_data/",
    "internal/",
    "runs/",
    ".venv/",
    ".tmp/",
    ".cache/",
    "dist/",
    "build/",
    "local_runs/",
)
FORBIDDEN_SUFFIXES = (".pdf", ".docx", ".doc", ".xlsx", ".png", ".jpg", ".jpeg", ".zip", ".whl", ".mp4", ".mp3", ".m4a")
# Reviewed documentation images (project-generated illustrations only) may be published here.
ALLOWED_IMAGE_PREFIX = "docs/assets/"
ALLOWED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".csv", ".yml", ".yaml", ".cfg", ".ini", ".cff", ""}
MAX_BYTES = 5 * 1024 * 1024
PERSONAL_PATH = re.compile(
    r"(?:\b[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"'`]+)|(?:/[a-z]/Users/[^/\s\"'`]+)|(?:/home/[^/\s\"'`]+)"
)
CREDENTIAL = re.compile(r"(?i)\b(api[_-]?key|secret|password|access[_-]?token)\b\s*[:=]\s*[\"'][^\"'\s]{8,}")


def published_files(root):
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Not a Git work tree or git unavailable: {exc}") from exc
    return sorted(p.decode("utf-8") for p in out.split(b"\0") if p)


def load_rules(path):
    if not path:
        return {"forbidden_prefixes": [], "forbidden_terms": []}
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "forbidden_prefixes": list(rules.get("forbidden_prefixes", [])),
        "forbidden_terms": [re.compile(t, re.IGNORECASE) for t in rules.get("forbidden_terms", [])],
    }


def check(root=ROOT, rules_path=None, strict=False, self_exempt=True):
    root = Path(root)
    rules = load_rules(rules_path)
    errors, warnings = [], []
    files = published_files(root)
    prefixes = FORBIDDEN_PREFIXES + tuple(rules["forbidden_prefixes"])
    for rel in files:
        path = root / rel
        if rel.startswith(prefixes):
            errors.append(f"{rel}: forbidden location")
        allowed_image = rel.startswith(ALLOWED_IMAGE_PREFIX) and rel.lower().endswith(ALLOWED_IMAGE_SUFFIXES)
        if rel.lower().endswith(FORBIDDEN_SUFFIXES) and not allowed_image:
            errors.append(f"{rel}: binary/original artefact not for publication")
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            errors.append(f"{rel}: larger than 5 MB")
        if not (path.is_file() and path.suffix.lower() in TEXT_SUFFIXES):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not (self_exempt and rel == "tools/check_public_tree.py"):
            if PERSONAL_PATH.search(text):
                errors.append(f"{rel}: personal absolute path")
            if CREDENTIAL.search(text):
                errors.append(f"{rel}: credential-like assignment")
        for term in rules["forbidden_terms"]:
            if term.search(text) or term.search(rel):
                errors.append(f"{rel}: confidential term matched ({term.pattern[:3]}...)")
                break
    if not any(name in files for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")):
        warnings.append("LICENSE: missing")
    if strict:
        errors.extend(warnings)
        warnings = []
    return files, errors, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="Git work tree to check")
    parser.add_argument("--rules", help="optional Git-ignored JSON with forbidden_prefixes / forbidden_terms")
    parser.add_argument("--strict", action="store_true", help="release mode: warnings are errors")
    args = parser.parse_args()
    files, errors, warnings = check(args.root, args.rules, args.strict)
    print(f"{len(files)} files would be published")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
