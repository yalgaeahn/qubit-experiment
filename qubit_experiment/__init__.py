"""Public package namespace for reusable qubit experiment components.

This namespace provides canonical imports for reusable modules while preserving
backward compatibility with legacy top-level imports in this repository.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["analysis", "experiments", "qpu_types", "helper_functions"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
