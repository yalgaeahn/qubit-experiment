from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _bootstrap_source() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root / "projects" / "2026_selectiveRIP" / "scripts" / "bootstrap_notebook.py"
    ).read_text(encoding="utf-8")


def test_bootstrap_notebook_adds_monorepo_package_root(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    project_root = repo / "projects" / "2026_selectiveRIP"
    script_path = project_root / "scripts" / "bootstrap_notebook.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(_bootstrap_source(), encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='tmp'\n", encoding="utf-8")
    (repo / "qubit_experiment").mkdir()

    baseline = list(sys.path)
    monkeypatch.setattr(sys, "path", baseline.copy())

    namespace = runpy.run_path(str(script_path))

    assert namespace["PROJECT_ROOT"] == project_root
    assert namespace["ADDED_PATHS"] == [project_root, repo]
    assert sys.path[:2] == [str(repo), str(project_root)]


def test_bootstrap_notebook_adds_sibling_package_root_for_split_layout(
    monkeypatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "2026_selectiveRIP"
    script_path = project_root / "scripts" / "bootstrap_notebook.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(_bootstrap_source(), encoding="utf-8")

    package_root = tmp_path / "qubit-experiment"
    (package_root / "qubit_experiment").mkdir(parents=True)
    (package_root / "pyproject.toml").write_text(
        "[project]\nname='qubit-experiment'\n", encoding="utf-8"
    )

    baseline = list(sys.path)
    monkeypatch.setattr(sys, "path", baseline.copy())

    namespace = runpy.run_path(str(script_path))

    assert namespace["PROJECT_ROOT"] == project_root
    assert namespace["ADDED_PATHS"] == [project_root, package_root]
    assert sys.path[:2] == [str(package_root), str(project_root)]
