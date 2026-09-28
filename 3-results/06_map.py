"""Map trend resolution at state and immediate-region geographic scales.

Panel A presents estimates from the state model. Panel B presents estimates from the
three-level immediate-region model. Both panels use the same outcome, 95% classification
criterion, and colour scale.

Run:
    .venv/bin/python 3-results/06_map.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, LABEL, model_dir, results_dir  # noqa: E402
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "ref"
MODEL = model_dir(ROOT)
FIGS = results_dir(ROOT) / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CACHE = PROC / "region_geometry.gpkg"
GREY = "#e3e3e3"
CALIBRI_DIR = Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts")
CALIBRI_FILES = [
    CALIBRI_DIR / "Calibri.ttf",
    CALIBRI_DIR / "Calibrib.ttf",
    CALIBRI_DIR / "Calibrii.ttf",
    CALIBRI_DIR / "Calibriz.ttf",
]
for font_path in CALIBRI_FILES:
    if not font_path.exists():
        raise FileNotFoundError(f"Required PPE figure font not found: {font_path}")
    font_manager.fontManager.addfont(font_path)

resolved_font = Path(
    font_manager.findfont("Calibri", fallback_to_default=False)
).resolve()
if resolved_font.name.lower() != "calibri.ttf":
    raise RuntimeError(f"Calibri regular did not resolve correctly: {resolved_font}")

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "Calibri",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.grid": False,
    }
)


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
    """Draw resolved units in colour and unresolved units in grey."""
    unresolved = gdf[~gdf.classified_95]
    resolved = gdf[gdf.classified_95]

    if len(unresolved):
        unresolved.plot(color=GREY, linewidth=0.15, edgecolor="white", ax=ax)
    if len(resolved):
        resolved.plot(
            column="change_pct",
            cmap=cmap,
            norm=norm,
            linewidth=0.15,
            edgecolor="white",
            ax=ax,
        )
    states_outline.boundary.plot(ax=ax, linewidth=0.45, color="0.25")
    ax.set_title(title, loc="left", fontweight="bold")
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

    # One symmetric scale permits direct comparison across geographic partitions.
    limit = float(
        np.percentile(
            np.abs(np.concatenate([states.change_pct, regions.change_pct])), 99
        )
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5))

    panel(
        axes[0],
        states,
        norm,
        cmap,
        states_geo,
        f"A. State model (n = 27)\n"
        f"{int(states.classified_95.sum())} of 27 trends resolved"
        f" ({states.classified_95.mean():.1%})",
    )
    panel(
        axes[1],
        regions,
        norm,
        cmap,
        states_geo,
        f"B. Three-level regional model (n = 510)\n"
        f"{int(regions.classified_95.sum())} of 510 trends resolved"
        f" ({regions.classified_95.mean():.1%})",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.02, aspect=40
    )
    cbar.set_label(
        f"Change in {LABEL} per livebirth, 2014–2024 (%)"
    )
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(
        FIGS / "figure1_map.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
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
