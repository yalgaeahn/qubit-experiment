"""
Self-contained example of complex resonator fitting (Qi, Qe) using lmfit.

This mirrors the complex S21 model used in the analysis module and can be
used as a reference for custom analysis or quick validation.

Usage:
  - Run directly to see synthetic data fitted and Qi/Qe printed.
  - Replace synthetic data with your measured frequencies (Hz) and complex S21.
"""

from __future__ import annotations

import numpy as np
import lmfit


def resonator_complex_realvec(f, f_0, Q, Q_e_real, Q_e_imag):
    """Return stacked real vector [Re(S21), Im(S21)] for fitting.

    Model: S21(f) = 1 - (Q/Qe)/(1 + 2j Q (f-f0)/f0), with Qe complex.
    """
    Qe = Q_e_real + 1j * Q_e_imag
    s21 = 1.0 - (Q * (1.0 / Qe)) / (1.0 + 2j * Q * (f - f_0) / f_0)
    return np.concatenate((np.real(s21), np.imag(s21)))


def compute_qi_qe_from_fit(result: lmfit.model.ModelResult) -> tuple[float, float, float]:
    """Compute (Q_loaded, Qi, Qe_external) from lmfit result."""
    Q_total = float(result.params["Q"].value)
    Qe_real = float(result.params["Q_e_real"].value)
    Qe_imag = float(result.params["Q_e_imag"].value)
    denom = Qe_real * Qe_real + Qe_imag * Qe_imag
    inv_Qe_real = Qe_real / denom if denom != 0 else 0.0
    inv_Q_total = 1.0 / Q_total if Q_total != 0 else 0.0
    inv_Qi = max(inv_Q_total - inv_Qe_real, 0.0)
    Qi = (1.0 / inv_Qi) if inv_Qi > 0 else float("inf")
    Qe_external = (1.0 / inv_Qe_real) if inv_Qe_real > 0 else float("inf")
    return Q_total, Qi, Qe_external


def demo_synthetic_fit():
    # Ground-truth parameters for synthetic data
    f0_true = 7.123e9
    Q_true = 2.5e4
    Qe_true = 1.8e4 - 0.9e4j

    f = np.linspace(f0_true - 2e6, f0_true + 2e6, 501)
    s21_true = 1.0 - (Q_true * (1.0 / Qe_true)) / (1.0 + 2j * Q_true * (f - f0_true) / f0_true)

    # Add a touch of complex noise
    rng = np.random.default_rng(1234)
    noise = (rng.normal(scale=2e-3, size=f.size) + 1j * rng.normal(scale=2e-3, size=f.size))
    s21_meas = s21_true + noise

    # Prepare real-valued vector for fitting
    data_vec = np.concatenate((np.real(s21_meas), np.imag(s21_meas)))

    # Build model and parameter hints
    model = lmfit.Model(resonator_complex_realvec)
    mag = np.abs(s21_meas)
    idx0 = int(np.argmin(mag))
    f0_guess = float(f[idx0])
    Q_guess = 1e4
    depth = float(max(1e-6, 1.0 - mag[idx0]))
    Qe_real_guess = max(10.0, Q_guess / depth)
    model.param_hints = {
        "f_0": {"value": f0_guess, "min": float(f[0]), "max": float(f[-1])},
        "Q": {"value": Q_guess, "min": 1.0, "max": 1e9},
        "Q_e_real": {"value": Qe_real_guess, "min": 1.0, "max": 1e12},
        "Q_e_imag": {"value": 0.0, "min": -1e12, "max": 1e12},
    }

    params = model.make_params()
    result = model.fit(data=data_vec, f=f, params=params)

    # Print fitted parameters
    f0 = result.params["f_0"].value
    Q = result.params["Q"].value
    Qe_r = result.params["Q_e_real"].value
    Qe_i = result.params["Q_e_imag"].value
    print("Fitted:")
    print(f"  f0 = {f0:.3f} Hz")
    print(f"  Q  = {Q:.3f}")
    print(f"  Qe = {Qe_r:.3f} + {Qe_i:.3f}j")

    Q_loaded, Qi, Qe_ext = compute_qi_qe_from_fit(result)
    print("Derived Qs:")
    print(f"  Q_loaded  = {Q_loaded:.3f}")
    print(f"  Q_internal (Qi) = {Qi:.3f}")
    print(f"  Q_external (Qe_ext) = {Qe_ext:.3f}")


if __name__ == "__main__":
    demo_synthetic_fit()

