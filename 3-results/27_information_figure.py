"""Figure 2 candidate: where recorded events are plentiful and where the slope is precise.

Two panels on the same 510 immediate regions (all-cause neonatal deaths, 2014-2024):
  A  recorded neonatal deaths per region-year (observed counts, log colour scale)
  B  posterior SD of the region's annual log-rate slope under the primary three-level model

Also writes the checks that the figure's caption and the Results rely on, computed directly
from the saved posterior (no model is refitted). P(slope < 0) is computed from the draws:
the stored `prob_direction` is the larger of the two sides and is NOT P(decline).

Run:  NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/27_information_figure.py
"""
import json
import sys
from pathlib import Path

import arviz as az
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, model_dir, results_dir  # noqa: E402

assert OUTCOME == "deaths_total"
MODEL, RES = model_dir(ROOT), results_dir(ROOT)
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)
BENCH = 2000  # conditional design benchmark, neonatal deaths per unit-year (8% decade decline, 80% power)

panel = pd.read_csv(ROOT / "data" / "processed" / "panel_region_year.csv")
deaths_year = (panel.groupby("rgi_id").deaths_total.sum() / panel.year.nunique()).rename("deaths_per_year")
geo = gpd.read_file(ROOT / "data" / "processed" / "region_geometry.gpkg")
slopes = pd.read_csv(MODEL / "slopes_rate.csv")

idata = az.from_netcdf(MODEL / "idata_rate.nc")
b = idata.posterior["b"].stack(sample=("chain", "draw")).values
regions = np.sort(panel.rgi_id.unique())
assert (slopes.rgi_id.values == regions).all()
df = pd.DataFrame({"rgi_id": regions, "b_mean": b.mean(1), "b_sd": b.std(1, ddof=1),
                   "p_decline": (b < 0).mean(1)})
df["classified_decline"] = np.percentile(b, 97.5, axis=1) < 0
df = df.merge(deaths_year, left_on="rgi_id", right_index=True).merge(geo[["rgi_id", "UF"]], on="rgi_id")
df.to_csv(RES / "tables" / "information_figure_data.csv", index=False)

NE = {"MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"}
med_d = df.deaths_per_year.median()
chk = {
    "regions": int(len(df)),
    "deaths_per_region_year_min_median_max": [float(df.deaths_per_year.min()), float(med_d), float(df.deaths_per_year.max())],
    "regions_at_or_above_benchmark": int((df.deaths_per_year >= BENCH).sum()),
    "regions_at_or_above_benchmark_ids": df.loc[df.deaths_per_year >= BENCH, ["rgi_id", "UF", "deaths_per_year", "classified_decline"]].to_dict("records"),
    "posterior_sd_min_median_max": [float(df.b_sd.min()), float(df.b_sd.median()), float(df.b_sd.max())],
    "posterior_sd_max_over_min": float(df.b_sd.max() / df.b_sd.min()),
    "classified_declines": int(df.classified_decline.sum()),
    "classified_declines_below_median_deaths": int((df.classified_decline & (df.deaths_per_year < med_d)).sum()),
    "classified_declines_in_northeast": int((df.classified_decline & df.UF.isin(NE)).sum()),
    "regions_in_northeast": int(df.UF.isin(NE).sum()),
    "p_decline_ge_0.975": int((df.p_decline >= 0.975).sum()),
    "p_decline_ge_0.95": int((df.p_decline >= 0.95).sum()),
    "p_decline_ge_0.90": int((df.p_decline >= 0.90).sum()),
    "p_decline_le_0.5": int((df.p_decline <= 0.5).sum()),
    "mean_posterior_mean_sign_disagrees_with_p_decline_gt_half": int(((df.b_mean < 0) != (df.p_decline > 0.5)).sum()),
    "stored_prob_direction_is_max_side": bool(np.allclose(slopes.prob_direction.values, np.maximum(df.p_decline, 1 - df.p_decline), atol=2e-3)),
    "expected_wrong_sign_if_p_decline_ge_0.975_regions_declared": float((1 - df.p_decline[df.p_decline >= 0.975]).sum()),
    "spearman_deaths_vs_posterior_sd": float(df[["deaths_per_year", "b_sd"]].corr(method="spearman").iloc[0, 1]),
}
json.dump(chk, open(RES / "tables" / "information_figure_checks.json", "w"), indent=2, default=float)

# ---- figure
m = geo.merge(df, on="rgi_id", suffixes=("", "_d"))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=300)
plt.rcParams["font.family"] = "DejaVu Sans"
panels = [
    ("deaths_per_year", "A. Recorded neonatal deaths per region-year", "viridis",
     LogNorm(vmin=max(1, df.deaths_per_year.min()), vmax=df.deaths_per_year.max()), "deaths per region-year (log scale)"),
    ("b_sd", "B. Posterior SD of the annual log-rate slope", "magma_r",
     matplotlib.colors.Normalize(vmin=df.b_sd.min(), vmax=df.b_sd.max()), "posterior SD (per year)"),
]
for ax, (col, title, cmap, norm, label) in zip(axes, panels):
    m.plot(column=col, cmap=cmap, norm=norm, linewidth=0.12, edgecolor="white", ax=ax, legend=False)
    geo.dissolve(by="UF").boundary.plot(ax=ax, color="black", linewidth=0.35)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.5, loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, shrink=0.8)
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    if col == "deaths_per_year":
        cb.ax.axvline(BENCH, color="red", linewidth=1.2)
        cb.ax.text(BENCH, 1.6, "2,000", color="red", fontsize=7.5, ha="center", transform=cb.ax.get_xaxis_transform())
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig2_information_map.{ext}", dpi=300, bbox_inches="tight")
print(json.dumps(chk, indent=2, default=float))
