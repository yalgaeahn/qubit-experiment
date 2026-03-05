"""Legacy compatibility package for `qpu_types` imports.

Canonical package path is `qubit_experiment.qpu_types`.
"""

from __future__ import annotations

from importlib import import_module

_pkg = import_module("qubit_experiment.qpu_types")
__path__ = _pkg.__path__
__all__ = getattr(_pkg, "__all__", [])

for _name in __all__:
    globals()[_name] = getattr(_pkg, _name)
