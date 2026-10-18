"""Replot the acquired confirmation traces without connecting to hardware."""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kps-mpl")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for label, title in [("precise_forward", "Drive on: forward"),
                     ("precise_reverse", "Drive on: reverse"),
                     ("drive_off", "Drive off")]:
    data = np.load(folder / f"{label}.npz")
    f, z = data["frequency_hz"], data["signal"]
    order = np.argsort(f)
    for ax, signal in zip(axes, [z.real, z.imag]):
        ax.plot((f[order] - 4.81303e9) / 1e6, signal[order], ".-", ms=3, lw=.8, label=title)
for ax, name in zip(axes, ["I", "Q"]):
    ax.set_ylabel(name + " (acquired units)")
    ax.grid(alpha=.25)
    ax.axvline(0, color="black", ls=":", lw=1)
axes[0].legend()
axes[0].set_title("Q3 spectroscopy: 4.81303 GHz\n10 us drive, -49.46 dBm instrument output, 1024 averages")
axes[-1].set_xlabel("Drive frequency offset from 4.81303 GHz (MHz)")
fig.tight_layout()
fig.savefig(folder / "q3_spectroscopy_confirmed.png", dpi=180)
