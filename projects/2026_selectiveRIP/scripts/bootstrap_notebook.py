"""Notebook bootstrap for project-local custom modules.

Run from notebooks with:

    %run ../scripts/bootstrap_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _is_qubit_experiment_repo(path: Path) -> bool:
    """Return whether `path` looks like the reusable package checkout."""

    return (path / "pyproject.toml").exists() and (path / "qubit_experiment").is_dir()


def _find_package_root(project_root: Path) -> Path | None:
    """Find the `qubit-experiment` checkout for in-repo or split layouts."""

    for candidate in project_root.parents:
        if _is_qubit_experiment_repo(candidate):
            return candidate

    sibling_package_root = project_root.parent / "qubit-experiment"
    if _is_qubit_experiment_repo(sibling_package_root):
        return sibling_package_root

    return None


def activate_project_paths() -> tuple[Path, list[Path]]:
    """Add the project root and any surrounding `qubit-experiment` repo to `sys.path`."""
    project_root = Path(__file__).resolve().parents[1]
    added: list[Path] = []

    candidates = [project_root]
    package_root = _find_package_root(project_root)
    if package_root is not None and package_root != project_root:
        candidates.append(package_root)

    for path in candidates:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added.append(path)

    return project_root, added


PROJECT_ROOT, ADDED_PATHS = activate_project_paths()
print(f"Project root: {PROJECT_ROOT}")
if ADDED_PATHS:
    print("Added to sys.path:")
    for path in ADDED_PATHS:
        print(f"  - {path}")
else:
    print("No new sys.path entries were needed.")
