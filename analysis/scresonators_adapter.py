from __future__ import annotations

"""Adapter to run scresonators fits on complex S21 arrays.

This keeps our analysis workflows decoupled from the external package while
providing a small, stable interface.
"""

from types import SimpleNamespace
from typing import Literal

import numpy as np


def fit_with_scresonators(
    fdata: np.ndarray,
    sdata: np.ndarray,
    method: Literal["DCM"] = "DCM",
    verbose: bool = False,
):
    """Run a scresonators fit and return an object exposing .params.

    Arguments:
        fdata: frequency array (same units as resonance frequency)
        sdata: complex S21 values for each frequency
        method: scresonators fit method (currently only 'DCM')
        verbose: print underlying fit report

    Returns:
        A SimpleNamespace with a single attribute `params` (lmfit.Parameters)
        compatible with downstream extraction in this repo.
    """
    # Lazy import so users without the dependency can still import our analysis
    import scresonators
    from scresonators.resonator import Resonator
    from scresonators.fit_methods import DCM

    if method != "DCM":
        raise ValueError("Only 'DCM' method is supported by this adapter right now.")

    fdata = np.asarray(fdata, dtype=float).ravel()
    sdata = np.asarray(sdata, dtype=np.complex128).ravel()

    res = Resonator(fdata=fdata, sdata=sdata)
    res.set_fitting_strategy(DCM())
    params = res.fit(verbose=verbose)

    # Expose a minimal ModelResult-like object with .params
    return SimpleNamespace(params=params)

