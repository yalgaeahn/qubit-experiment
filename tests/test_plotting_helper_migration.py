from __future__ import annotations

from importlib import import_module
from pathlib import Path

ANALYSIS_DIR = Path(import_module("qubit_experiment.analysis").__file__).resolve().parent


def test_analysis_no_external_plotting_helper_imports() -> None:
    offenders: list[tuple[str, int, str]] = []
    needle = "laboneq_applications.analysis.plotting_helpers"

    for path in sorted(ANALYSIS_DIR.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if needle in line:
                offenders.append((path.name, lineno, line.strip()))

    assert offenders == [], f"External plotting helper import remains: {offenders}"
