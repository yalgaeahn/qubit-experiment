from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [
    REPO_ROOT / "qubit_experiment",
    REPO_ROOT / "tests",
    REPO_ROOT / "example_helpers",
]
BANNED_IMPORT_ROOTS = {
    "analysis",
    "experiments",
    "qpu_types",
    "helper_functions",
    "helper",
    "operations",
    "custom_pulse_library",
}
SMOKE_MODULES = [
    "qubit_experiment.analysis.plot_theme",
    "qubit_experiment.experiments.iq_cloud_common",
    "qubit_experiment.helper",
    "qubit_experiment.custom_pulse_library",
]
REMOVED_MODULES = [
    "analysis.plot_theme",
    "experiments.iq_cloud_common",
    "qpu_types.transmon",
    "helper_functions.helper",
    "helper",
    "operations",
    "custom_pulse_library",
]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for base in SCAN_DIRS:
        files.extend(sorted(base.rglob("*.py")))
    return files


def _legacy_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            root = node.module.split(".", 1)[0]
            if root in BANNED_IMPORT_ROOTS:
                offenders.append(node.module)

    return offenders


def test_no_top_level_legacy_imports_remain() -> None:
    offenders: list[tuple[str, list[str]]] = []

    for path in _python_files():
        bad_imports = _legacy_imports(path)
        if bad_imports:
            offenders.append((str(path.relative_to(REPO_ROOT)), bad_imports))

    assert offenders == []


def test_canonical_smoke_imports() -> None:
    for module_name in SMOKE_MODULES:
        assert import_module(module_name) is not None


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_top_level_modules_are_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module(module_name)
