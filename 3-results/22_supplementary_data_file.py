"""Supplementary Data File 1: per-region slopes from the primary three-level fit.

One row per immediate region: code and state, livebirths and outcome deaths over the
decade, deaths per unit-year, the slope with its posterior SD and 95% CrI, the implied
percentage change per decade, and the classification. Drawn from slopes_rate.csv of the
selected outcome, so every value shown in Figure 1B is also reported numerically.

Run:
    .venv/bin/python 3-results/22_supplementary_data_file.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, model_dir, results_dir  # noqa: E402

PROC = ROOT / "data" / "processed"
MODEL = model_dir(ROOT)
OUT = results_dir(ROOT) / "tables"


def main():
    panel = pd.read_csv(PROC / "panel_region_year.csv")
    years = panel.year.nunique()
    pooled = (panel.groupby("rgi_id")
              .agg(UF=("UF", "first"), births=("births", "sum"), deaths=(OUTCOME, "sum"))
              .reset_index())
    slopes = pd.read_csv(MODEL / "slopes_rate.csv")
    df = pooled.merge(slopes, on="rgi_id", how="inner")
    assert len(df) == 510, len(df)
    label = "avoidable_deaths" if OUTCOME == "avoidable" else "neonatal_deaths"
    out = pd.DataFrame({
        "rgi_id": df.rgi_id,
        "UF": df.UF,
        "births_2014_2024": df.births.astype(int),
        f"{label}_2014_2024": df.deaths.astype(int),
        f"{label}_per_unit_year": (df.deaths / years).round(1),
        "slope_per_year": df.b_mean.round(5),
        "posterior_sd": df.b_sd.round(5),
        "lo95": df.b_lo95.round(5),
        "hi95": df.b_hi95.round(5),
        "pct_change_per_decade": (100 * (np.exp(10 * df.b_mean) - 1)).round(1),
        "classified_95": df.classified_95.astype(bool),
    }).sort_values("rgi_id")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "supplementary_data_S11_region_slopes.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}: {len(out)} rows, {int(out.classified_95.sum())} classified")


if __name__ == "__main__":
    main()
