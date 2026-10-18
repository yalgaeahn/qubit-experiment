"""Replot only completed acquisitions; no instrument connection or refitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
metrics = json.loads((folder / "03_low_power.fit.json").read_text())
reference = metrics["reference_frequency_hz"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained", sharex=True)
for i, point in enumerate(metrics["power_points"]):
    with np.load(folder / f"03_low_power_{i}.npz") as saved:
        f = saved["frequencies_hz"]
        raw = saved["raw_complex"][0]
    x = (f - reference) / 1e6
    p = np.asarray(point["parameters"])
    response = (p[2] + 1j*p[3]) + (p[4] + 1j*p[5])/(1 + 2j*(x-p[0])/np.exp(p[1]))
    fitted = response * np.max(np.abs(raw)) * np.exp(-2j*np.pi*(f-reference)*point["delay_s"])
    normalization = np.median(np.abs(fitted[np.abs(x) > 1.5]))
    correction = np.exp(2j*np.pi*(f-reference)*point["delay_s"])
    color = "#167568" if point["status"] == "ok" else "#b06b20"
    axes[0, i].plot(x, np.abs(raw)/normalization, ".", ms=2, alpha=.35, color=color, rasterized=True)
    axes[0, i].plot(x, np.abs(fitted)/normalization, color="black", lw=1.6, label="complex reflection fit")
    axes[1, i].plot(x, np.angle(raw*correction), ".", ms=2, alpha=.35, color=color, rasterized=True)
    axes[1, i].plot(x, np.angle(fitted*correction), color="black", lw=1.6)
    status = "fit checks passed" if point["status"] == "ok" else "uncertainty / window checks failed"
    axes[0, i].set_title(f"{point['output_power_dbm']:.1f} dBm (port)\n{status}\n"
                         f"linewidth = {point['linewidth_hz']/1e6:.3f} ± {point['linewidth_stderr_hz']/1e6:.3f} MHz",
                         fontsize=10, color=color)
    axes[1, i].set_xlabel("Frequency − 7.3452 GHz (MHz)")
    for ax in axes[:, i]:
        ax.grid(alpha=.18)
axes[0, 0].set_ylabel("Normalized magnitude")
axes[1, 0].set_ylabel("Delay-corrected phase (rad)")
axes[0, 2].legend(loc="lower right", fontsize=8)
fig.suptitle("q2 = qubits[1] · CW reflection · completed 16 µs acquisitions\n"
             "Low-power plateau is NOT established; error bars are conditional statistical uncertainties", fontsize=13)
fig.savefig(folder / "q2_power_comparison.png", dpi=160)
plt.close(fig)
print(folder / "q2_power_comparison.png")
for p in metrics["power_points"]:
    print({k:p[k] for k in ["output_power_dbm", "fr_hz", "fr_stderr_hz", "q_loaded", "q_loaded_stderr", "linewidth_hz", "linewidth_stderr_hz", "energy_decay_time_s", "status"]})
