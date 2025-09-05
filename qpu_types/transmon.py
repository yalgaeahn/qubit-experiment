"""Backward-compatibility shim for transmon imports.

Allows `from qpu_types.transmon import TransmonQubit` to work after package
reorganization.
"""

from .Transmon.transmon import TransmonQubit, TransmonQubitParameters  # noqa: F401

__all__ = ["TransmonQubit", "TransmonQubitParameters"]

