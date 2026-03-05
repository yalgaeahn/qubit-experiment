"""Legacy compatibility package for `helper_functions` imports.

Canonical package path is `qubit_experiment.helper_functions`.
"""

from __future__ import annotations

from importlib import import_module

_pkg = import_module("qubit_experiment.helper_functions")
__path__ = _pkg.__path__
__all__ = getattr(_pkg, "__all__", [])

for _name in __all__:
    globals()[_name] = getattr(_pkg, _name)
