"""Publication hygiene checks (generic rules only)."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("check_public_tree", ROOT / "tools/check_public_tree.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def in_git_tree(path):
    if shutil.which("git") is None:
        return False
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, capture_output=True, text=True)
    return probe.returncode == 0 and probe.stdout.strip() == "true"


@pytest.mark.skipif(not in_git_tree(ROOT), reason="requires a Git work tree (not available in sdist/wheel installs)")
def test_work_tree_has_no_generic_violations():
    files, errors, _ = load_tool().check(ROOT)
    assert files
    assert errors == []


def test_personal_path_pattern_detects_examples():
    module = load_tool()
    # Built at run time so this source file itself contains no personal-path literal.
    assert module.PERSONAL_PATH.search("D:" + "\\Users\\someone\\project")
    assert module.PERSONAL_PATH.search("/ho" + "me/someone/project")
    assert not module.PERSONAL_PATH.search("data/public/arep21_r404a_65f/data.csv")


@pytest.mark.skipif(shutil.which("git") is None, reason="requires git")
def test_rules_file_blocks_terms_and_prefixes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "secret_dir").mkdir(parents=True)
    (repo / "README.md").write_text("mentions codeword-xyz here\n", encoding="utf-8")
    (repo / "secret_dir" / "a.txt").write_text("plain\n", encoding="utf-8")
    (repo / "LICENSE").write_text("MIT\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps({"forbidden_prefixes": ["secret_dir/"], "forbidden_terms": ["codeword-xyz"]}), encoding="utf-8"
    )
    _, errors, warnings = load_tool().check(repo, rules, strict=True)
    assert any("README.md" in e and "confidential" in e for e in errors)
    assert any("secret_dir/a.txt" in e for e in errors)
    assert warnings == []
