"""Compatibility shim for operations.

Re-exports fixed-transmon operations under the legacy `TransmonOperations`
name so existing notebooks continue to work.
"""

from qubit_experiment.qpu_types.fixed_transmon.operations import (
    FixedTransmonOperations,
    create_pulse,
)

TransmonOperations = FixedTransmonOperations

__all__ = ["TransmonOperations", "create_pulse", "FixedTransmonOperations"]
