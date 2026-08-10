"""Compute the published quantities that no other script produced.

An audit found five numbers in the manuscript that were correct but had no committed
code behind them: they had been derived ad hoc and typed in. That fails the standard this
project sets for itself, so each is computed here, written to a JSON the notebooks assert
against, and thereby brought into the audit chain.

    aggregation_shift        the licence for reading the ladder: how far a coarser unit's
                             slope departs from the death-weighted blend of its parts
    state_multiplicity       Benjamini-Hochberg and Bonferroni over the 27 state contrasts
    region_departures        region departures from the national drift under each of the
                             two variance functions
    state_level_from_region  states resolved when read off the three-level region model
    rho_below_half           posterior mass of the spatial share below 0.5
    unpooled_sign_split      how many unpooled classifications are rising rather than
                             falling, at each rung of the ladder. This one is not a
                             tidy-up: the manuscript previously said every resolved unit
                             at every level was improving, which is true of the
                             hierarchical fits and false of the unpooled column printed
                             beside them.

Run:
    .venv/bin/python 3-results/17_orphan_numbers.py
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = ROOT / "2-model"
OUT = ROOT / "3-results"

MACRO = {
    "AC": "North", "AM": "North", "AP": "North", "PA": "North", "RO": "North",
    "RR": "North", "TO": "North", "AL": "Northeast", "BA": "Northeast",
    "CE": "Northeast", "MA": "Northeast", "PB": "Northeast", "PE": "Northeast",
    "PI": "Northeast", "RN": "Northeast", "SE": "Northeast", "DF": "Centre-West",
    "GO": "Centre-West", "MS": "Centre-West", "MT": "Centre-West", "ES": "Southeast",
    "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast", "PR": "South",
    "RS": "South", "SC": "South",
}


def unpooled(frame, key, mean_year):
    """Quasi-Poisson slope per unit, t on n-2 residual df, the study's standard."""
    g = frame.groupby([key, "year"], as_index=False)[["avoidable", "births"]].sum()
    g["t"] = g.year - mean_year
    g = g[g.births > 0]
    rows = []
    for unit, x in g.groupby(key):
        if x.avoidable.sum() == 0:
            continue
        try:
            f = sm.GLM(
                x.avoidable, sm.add_constant(x[["t"]]),
                family=sm.families.Poisson(), offset=np.log(x.births),
            ).fit(scale="X2")
            b, se = float(f.params["t"]), float(f.bse["t"])
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
            continue
        crit = stats.t.ppf(0.975, len(x) - 2)
        rows.append({"unit": unit, "b": b, "se": se, "classified": abs(b) > crit * se})
    return pd.DataFrame(rows)


def aggregation_shift(panel, mean_year):
    """Region-to-state, and the municipality-to-region rung the audit found missing."""
    out = {}
    for coarse, fine, label in [("UF", "rgi_id", "region_to_state"),
                                ("rgi_id", "CODMUNRES", "municipality_to_region")]:
        fine_fits = unpooled(panel, fine, mean_year).set_index("unit").b
        weights = panel.groupby(fine).avoidable.sum()
        coarse_fits = unpooled(panel, coarse, mean_year).set_index("unit").b
        gaps = []
        for c, g in panel.groupby(coarse):
            ids = [u for u in g[fine].unique() if u in fine_fits.index]
            if not ids or c not in coarse_fits.index:
                continue
            blend = np.average([fine_fits[u] for u in ids], weights=[weights[u] for u in ids])
            gaps.append(abs(coarse_fits[c] - blend))
        out[label] = {"median": round(float(np.median(gaps)), 5),
                      "max": round(float(np.max(gaps)), 5), "units": len(gaps)}
    return out


def state_multiplicity():
    s = pd.read_csv(OUT / "tables" / "state_dispersion.csv")
    p = np.sort(s.p.values)
    n = len(p)
    passing = p <= np.arange(1, n + 1) / n * 0.05
    k = int(np.max(np.where(passing)[0]) + 1) if passing.any() else 0
    return {
        "raw_p_below_0.05": int((s.p < 0.05).sum()),
        "bh_survivors": k,
        "bh_states": sorted(s.nsmallest(k, "p").UF.tolist()) if k else [],
        "bonferroni_survivors": int((s.p < 0.05 / n).sum()),
        "bonferroni_states": sorted(s.loc[s.p < 0.05 / n, "UF"].tolist()),
    }


def departures(path):
    idata = az.from_netcdf(path)
    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    mu = idata.posterior["mu_b"].stack(sample=("chain", "draw")).values
    d = b - mu[None, :]
    lo, hi = np.quantile(d, 0.025, axis=1), np.quantile(d, 0.975, axis=1)
    return int(((lo > 0) | (hi < 0)).sum())


def states_from_region_model():
    idata = az.from_netcdf(MODEL / "idata_rate.nc")
    s = (idata.posterior["mu_b"] + idata.posterior["b_state"]).stack(
        sample=("chain", "draw")
    ).values
    lo, hi = np.quantile(s, 0.025, axis=1), np.quantile(s, 0.975, axis=1)
    return int(((lo > 0) | (hi < 0)).sum())


def rho_below_half():
    idata = az.from_netcdf(MODEL / "idata_spatial_bym2.nc")
    r = idata.posterior["rho"].values.flatten()
    return round(float((r < 0.5).mean()), 3)


def unpooled_sign_split(panel, mean_year):
    """Rising versus falling among unpooled classifications, at each ladder rung."""
    panel = panel.copy()
    panel["macro"] = panel.UF.map(MACRO)
    out = {}
    for label, key in [("municipality", "CODMUNRES"), ("immediate region", "rgi_id"),
                       ("state", "UF"), ("macro-region", "macro")]:
        fits = unpooled(panel, key, mean_year)
        cls = fits[fits.classified]
        out[label] = {
            "classified": int(len(cls)),
            "falling": int((cls.b < 0).sum()),
            "rising": int((cls.b > 0).sum()),
        }
    return out


def main():
    panel = pd.read_csv(PROC / "panel_muni_year.csv").dropna(subset=["rgi_id"]).copy()
    panel["rgi_id"] = panel.rgi_id.astype("int64")
    mean_year = panel.year.mean()

    result = {
        "aggregation_shift": aggregation_shift(panel, mean_year),
        "state_multiplicity": state_multiplicity(),
        "region_departures": {
            "canonical_global_dispersion": departures(MODEL / "idata_rate.nc"),
            "exposure_dependent_dispersion": departures(MODEL / "idata_disp_exposure.nc"),
        },
        "states_resolved_from_region_model": states_from_region_model(),
        "rho_posterior_below_0.5": rho_below_half(),
        "unpooled_sign_split": unpooled_sign_split(panel, mean_year),
    }
    (OUT / "ORPHAN_NUMBERS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
