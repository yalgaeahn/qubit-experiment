"""Transmon aliases for fixed-transmon types."""

from .fixed_transmon.qubit_types import (
    FixedTransmonQubit,
    FixedTransmonQubitParameters,
)

TransmonQubit = FixedTransmonQubit
TransmonQubitParameters = FixedTransmonQubitParameters

__all__ = ["TransmonQubit", "TransmonQubitParameters"]
