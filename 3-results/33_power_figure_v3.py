"""Figure 1 for the manuscript: power of the unpooled procedure against expected deaths per unit-year.

Re-plots the stored curve (power_threshold_curve.csv) with labels that describe the scenarios as true
declines per decade (8.0% = the hierarchical mean slope; 14.3% = the median 95% interval half-width of
the primary regional model; 20%). Points are binned simulated power, lines the analytic calculation.

Run:  NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/33_power_figure_v3.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, results_dir  # noqa: E402

assert OUTCOME == "deaths_total"
RES = results_dir(ROOT)
curve = pd.read_csv(RES / "tables" / "power_threshold_curve.csv")
info = json.load(open(RES / "POWER_THRESHOLD.json"))
thr, obs = info["thresholds"], info["observed_deaths_per_unit_year"]
series = [("national_8pct", "8.0% per decade (hierarchical mean slope)", "#2a6fd6"),
          ("mde_14pct", "14.3% per decade (median 95% interval half-width)", "#e8622c"),
          ("decline_20pct", "20% per decade", "#1fa67a")]
fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=300)
q = obs["quintile_medians_immediate_region"]
ax.axvspan(q[0], q[-1], color="#e8e6e1", zorder=0)
ax.text(np.sqrt(q[0] * q[-1]), 0.97, "observed immediate regions\n(quintile medians)", ha="center", va="top", fontsize=8, color="#555555")
for key, label, colr in series:
    c = curve[curve.effect == key].sort_values("deaths_per_unit_year")
    ax.plot(c.deaths_per_unit_year, c.analytic_power, color=colr, lw=2, label=label, zorder=3)
    ax.scatter(c.deaths_per_unit_year, c.power, s=16, facecolor="white", edgecolor=colr, lw=1.2, zorder=4)
    a80 = thr[key]["analytic_deaths_per_unit_year_for_80pct_power"]
    ax.scatter([a80], [0.8], marker="D", s=26, color=colr, edgecolor="white", lw=0.8, zorder=6)
    ax.text(a80 * 1.10, 0.735, f"{a80:,.0f}", fontsize=8.5, color=colr, ha="left", va="top", fontweight="bold")
ax.axhline(0.8, color="#bbbbbb", lw=0.8, ls="--"); ax.text(2.1, 0.815, "80% power", fontsize=8, color="#555555")
ax.axhline(0.05, color="#999999", lw=0.8, ls="--"); ax.text(14000, 0.065, "5% false-declaration rate", fontsize=8, color="#555555", ha="right")
for x, lab in ((obs["state"], "median state"), (obs["macro-region"], "median macro-region")):
    ax.axvline(x, color="#999999", lw=0.8, ls=":"); ax.text(x * 1.04, 0.12, lab, rotation=90, fontsize=7.5, color="#555555", va="bottom")
ax.set_xscale("log"); ax.set_xlim(2, 20000); ax.set_ylim(0, 1.0)
ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200", "500", "1,000", "2,000", "5,000", "10,000"])
ax.set_xlabel("Expected neonatal deaths per unit-year (all causes)"); ax.set_ylabel("Power of the unpooled fit of an 11-year series")
ax.legend(title="True change", loc="center left", bbox_to_anchor=(0.01, 0.45), fontsize=8, title_fontsize=8.5, frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(RES / "figures" / f"fig1_power_curve.{ext}", dpi=300, bbox_inches="tight")
print("ok")
