from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

ANALYSIS_DIR = Path(import_module("qubit_experiment.analysis").__file__).resolve().parent
PLOT_CALL_TARGETS = {("plt", "subplots"), ("plt", "figure"), ("plt", "show")}


def _function_has_plot_call(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Attribute):
            value = subnode.func.value
            if isinstance(value, ast.Name) and (value.id, subnode.func.attr) in PLOT_CALL_TARGETS:
                return True
    return False


def _function_has_theme_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "with_plot_theme":
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            if dec.func.id == "with_plot_theme":
                return True
    return False


def test_all_analysis_plot_functions_use_unified_theme_decorator() -> None:
    checked = 0
    missing: list[tuple[str, str, int]] = []

    for path in sorted(ANALYSIS_DIR.glob("*.py")):
        if path.name in {"plot_theme.py", "__init__.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _function_has_plot_call(node):
                continue
            checked += 1
            if not _function_has_theme_decorator(node):
                missing.append((path.name, node.name, node.lineno))

    assert checked > 0
    assert missing == [], f"Missing @with_plot_theme on: {missing}"
