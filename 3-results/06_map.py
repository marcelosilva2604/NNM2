"""The figure the paper is built on: the same country, the same decade, two zoom levels.

Panel A shows the 27 states, the level at which Brazil and international comparisons
report neonatal mortality. Most states resolve, and the map reads as a country that knows
what is happening to it.

Panel B shows the 510 immediate regions, closer to the level at which care is organised
and delivered. Most of the map goes grey.

Both panels use the same likelihood, the same estimand, the same 95% criterion and the
same colour scale. Only the unit of analysis changes.

Run:
    .venv/bin/python 3-results/06_map.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "ref"
MODEL = ROOT / "2-model"
FIGS = ROOT / "3-results" / "figures"

CACHE = PROC / "region_geometry.gpkg"
GREY = "#e3e3e3"


def region_geometry():
    """Dissolve the municipal mesh into immediate regions, cached after the first build."""
    if CACHE.exists():
        return gpd.read_file(CACHE)

    munis = gpd.read_file(REF / "br_municipios.geojson")
    code_col = next(
        c
        for c in munis.columns
        if munis[c].astype(str).str.fullmatch(r"\d{7}").fillna(False).mean() > 0.9
    )
    munis["cod7"] = munis[code_col].astype("int64")
    if (~munis.geometry.is_valid).any():
        munis["geometry"] = munis.geometry.make_valid()

    xwalk = pd.read_csv(REF / "muni_regiao_imediata.csv")[["cod7", "rgi_id", "UF"]]
    regions = (
        munis.merge(xwalk, on="cod7", how="inner")
        .dissolve(by="rgi_id", aggfunc={"UF": "first"})
        .reset_index()
    )
    regions.to_file(CACHE, driver="GPKG")
    return regions


def per_decade(slope):
    """Convert the log-scale annual slope into a percentage change over the decade."""
    return 100 * (np.exp(slope * 10) - 1)


def panel(ax, gdf, norm, cmap, states_outline, title):
    """Draw one zoom level: resolved units in colour, unresolved in grey."""
    unresolved = gdf[~gdf.classified_95]
    resolved = gdf[gdf.classified_95]

    if len(unresolved):
        unresolved.plot(color=GREY, linewidth=0.08, edgecolor="white", ax=ax)
    if len(resolved):
        resolved.plot(
            column="change_pct",
            cmap=cmap,
            norm=norm,
            linewidth=0.08,
            edgecolor="white",
            ax=ax,
        )
    states_outline.boundary.plot(ax=ax, linewidth=0.3, color="0.3")
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_axis_off()


def main():
    states_geo = gpd.read_file(REF / "br_states.geojson")[["SIGLA", "geometry"]].rename(
        columns={"SIGLA": "UF"}
    )

    state_slopes = pd.read_csv(MODEL / "slopes_state.csv")
    region_slopes = pd.read_csv(MODEL / "slopes_rate.csv")

    states = states_geo.merge(state_slopes, on="UF")
    regions = region_geometry().merge(region_slopes, on="rgi_id")
    states["change_pct"] = per_decade(states.b_mean)
    regions["change_pct"] = per_decade(regions.b_mean)

    # One symmetric scale for both panels, so the two zoom levels are directly comparable.
    limit = float(
        np.percentile(
            np.abs(np.concatenate([states.change_pct, regions.change_pct])), 99
        )
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))

    panel(
        axes[0],
        states,
        norm,
        cmap,
        states_geo,
        f"A. By state (n = 27)\n{int(states.classified_95.sum())} of 27 resolved"
        f" ({states.classified_95.mean():.0%})",
    )
    panel(
        axes[1],
        regions,
        norm,
        cmap,
        states_geo,
        f"B. By immediate region (n = 510)\n"
        f"{int(regions.classified_95.sum())} of 510 resolved"
        f" ({regions.classified_95.mean():.0%})",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.02, aspect=40
    )
    cbar.set_label(
        "Change in avoidable neonatal deaths per birth, 2014-2024 (%)", fontsize=7.5
    )
    cbar.ax.tick_params(labelsize=7)

    # Grey is a category, not missing data, and the legend has to say which.
    axes[1].scatter(
        [],
        [],
        marker="s",
        s=30,
        color=GREY,
        edgecolor="0.55",
        linewidth=0.4,
        label="No determination",
    )
    axes[1].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.02))

    fig.savefig(FIGS / "figure1_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"states  resolved {int(states.classified_95.sum())}/27"
        f" | rising {int((states.classified_95 & (states.b_mean > 0)).sum())}"
    )
    print(
        f"regions resolved {int(regions.classified_95.sum())}/510"
        f" | rising {int((regions.classified_95 & (regions.b_mean > 0)).sum())}"
    )
    print(f"shared colour scale capped at +-{limit:.1f}% per decade")


if __name__ == "__main__":
    main()
