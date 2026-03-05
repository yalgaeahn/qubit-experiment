"""Backward-compatibility shim for transmon imports.

Allows `from qpu_types.transmon import TransmonQubit` to work after package
reorganization.
"""

from .fixed_transmon.qubit_types import (
    FixedTransmonQubit,
    FixedTransmonQubitParameters,
)

TransmonQubit = FixedTransmonQubit
TransmonQubitParameters = FixedTransmonQubitParameters

__all__ = ["TransmonQubit", "TransmonQubitParameters"]
