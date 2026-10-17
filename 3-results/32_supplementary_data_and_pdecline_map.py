"""Supplementary data file (per region) and eFigure 2 (posterior probability of decline).

Merges the per-region slope summaries, recorded deaths, P(slope < 0) computed from the draws, and the
classification under the four slope priors. eFigure 2 maps P(decline) on a fixed 0-1 scale with a
neutral midpoint at 0.5, labelled as the posterior probability of decline under the primary model.

Run:  NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/32_supplementary_data_and_pdecline_map.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, results_dir  # noqa: E402

assert OUTCOME == "deaths_total"
RES = results_dir(ROOT)
panel = pd.read_csv(ROOT / "data" / "processed" / "panel_region_year.csv")
info = pd.read_csv(RES / "tables" / "information_figure_data.csv")
spec = pd.read_csv(RES / "audit_2026-10-17" / "region_by_specification.csv")
tot = panel.groupby("rgi_id").agg(births_2014_2024=("births", "sum"), neonatal_deaths_2014_2024=("deaths_total", "sum")).reset_index()
nm = panel[["rgi_id", "rgi_nome"]].drop_duplicates()
d = (tot.merge(nm, on="rgi_id").merge(info[["rgi_id", "UF", "deaths_per_year", "b_mean", "b_sd", "p_decline", "classified_decline"]], on="rgi_id")
       .merge(spec[["rgi_id"] + [f"{k}_classified" for k in ("primary_three_level", "nopool", "exchangeable", "bym2_scaled")]], on="rgi_id"))
d["slope_per_year"] = d.b_mean
d["lo95"] = d.b_mean - 1.96 * d.b_sd  # normal approximation only for the file; exact intervals are in slopes_rate.csv
ex = pd.read_csv(ROOT / "2-model" / "allcause" / "slopes_rate.csv")[["rgi_id", "b_lo95", "b_hi95"]]
d = d.merge(ex, on="rgi_id").drop(columns=["lo95", "b_mean"]).rename(columns={"b_lo95": "slope_lo95", "b_hi95": "slope_hi95", "b_sd": "slope_posterior_sd", "p_decline": "posterior_probability_of_decline"})
d = d[["rgi_id", "rgi_nome", "UF", "births_2014_2024", "neonatal_deaths_2014_2024", "deaths_per_year", "slope_per_year", "slope_posterior_sd", "slope_lo95", "slope_hi95", "posterior_probability_of_decline",
       "primary_three_level_classified", "nopool_classified", "exchangeable_classified", "bym2_scaled_classified"]]
d.to_csv(RES / "tables" / "supplementary_data_regions.csv", index=False, float_format="%.6g")
print(d.shape)

geo = gpd.read_file(ROOT / "data" / "processed" / "region_geometry.gpkg").merge(d[["rgi_id", "posterior_probability_of_decline"]], on="rgi_id")
fig, ax = plt.subplots(figsize=(6.2, 6.4), dpi=300)
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
geo.plot(column="posterior_probability_of_decline", cmap="BrBG", norm=norm, linewidth=0.12, edgecolor="white", ax=ax)
gpd.read_file(ROOT / "data" / "processed" / "region_geometry.gpkg").dissolve(by="UF").boundary.plot(ax=ax, color="black", linewidth=0.35)
ax.set_axis_off()
sm = plt.cm.ScalarMappable(cmap="BrBG", norm=norm)
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, shrink=0.8)
cb.set_label("posterior probability of decline under the primary model", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(RES / "figures" / f"efig2_probability_of_decline.{ext}", dpi=300, bbox_inches="tight")
print("figure written")
