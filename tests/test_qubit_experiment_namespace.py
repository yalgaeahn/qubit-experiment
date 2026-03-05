from __future__ import annotations

from importlib import import_module
from pathlib import Path


def test_legacy_analysis_submodule_resolves_to_canonical_source() -> None:
    legacy = import_module("analysis.plot_theme")
    canonical = import_module("qubit_experiment.analysis.plot_theme")
    assert Path(legacy.__file__).resolve() == Path(canonical.__file__).resolve()


def test_helper_alias_module_is_importable() -> None:
    helper_alias = import_module("qubit_experiment.helper")
    helper_legacy = import_module("helper")
    assert hasattr(helper_alias, "load_qubit_parameters")
    assert hasattr(helper_legacy, "load_qubit_parameters")
