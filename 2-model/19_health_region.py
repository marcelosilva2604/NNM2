"""The rung the paper was missing: the geography in which SUS care is actually planned.

The ladder in 09_aggregation_ladder.py uses IBGE's immediate region as its intermediate
rung. That is an economic geography. A reviewer's obvious objection, and one the paper
itself declares as a limitation, is that the health system does not plan in it: SUS care is
planned in *regiões de saúde*, pactuated by each state's Comissão Intergestores Regional.

The two partitions are of similar granularity but they are not nested in each other and
they cross: 433 health regions against 510 immediate regions. So this is not a fifth step
on the same ladder, it is a competing partition at the same scale, and the question it
answers is sharper than the ladder's. If resolving power were an artefact of choosing an
administratively irrelevant geography, moving to the planning geography would fix it.

Everything except the partition is held identical to the ladder: same likelihood, same
two-level hierarchy (national plus unit), same offset, same centring, same sampler
settings, same 95% criterion, and the same unpooled quasi-Poisson comparison referred to t
on n-2 residual degrees of freedom. Only the map changes.

    Crosswalk: DATASUS territorial base, December 2024 vintage,
    ftp://ftp.datasus.gov.br/territorio/tabelas/2024/, tables rl_municip_regsaud and
    tb_regsaud. The extracted crosswalk is versioned in this repository as
    data/ref/muni_regiao_saude.csv, with its provenance in the companion PROVENANCE.md, so
    running this script does not require downloading the archive.

A caveat this script records rather than hides: health regions were repactuated during the
study decade in some states, most visibly Ceará and Espírito Santo, so a single contemporary
partition is applied retrospectively to the whole period. That is the same treatment the
IBGE immediate regions receive (2017 division), and it is the standard choice when the
estimand is a within-unit trend, but it is an assumption and is reported as one.

Run:
    .venv/bin/python 2-model/19_health_region.py
"""

import json
import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "ref"
OUT = ROOT / "2-model"
RES = ROOT / "3-results"

# Identical to 09_aggregation_ladder.py. Do not tune these independently: the whole point
# of this fit is that only the partition differs from the immediate-region rung.
SEED = 20260803
DRAWS = 1500
TUNE = 2500
CHAINS = 4
TARGET_ACCEPT = 0.95


def panel():
    """Aggregate the municipality panel to health regions, on the ladder's conventions."""
    muni = pd.read_csv(PROC / "panel_muni_year.csv")
    xwalk = pd.read_csv(REF / "muni_regiao_saude.csv", dtype={"cod6": str, "regsaud_id": str})

    muni["cod6"] = muni.CODMUNRES.astype(str).str[:6]
    merged = muni.merge(xwalk, on="cod6", how="left")

    unmapped = merged[merged.regsaud_id.isna()]
    provenance = {
        "municipality_codes_unmapped": int(unmapped.cod6.nunique()),
        "municipality_years_unmapped": int(len(unmapped)),
        "births_unmapped": int(unmapped.births.sum()),
        "avoidable_deaths_unmapped": int(unmapped.avoidable.sum()),
    }

    merged = merged.dropna(subset=["regsaud_id"])
    g = merged.groupby(["regsaud_id", "year"], as_index=False)[["avoidable", "births"]].sum()
    g = g.rename(columns={"regsaud_id": "unit"})
    g = g[g.births > 0].copy()
    # Centring uses the municipality panel's mean year, exactly as the ladder does, so the
    # intercept means the same thing at every rung.
    g["t"] = g.year - muni.year.mean()
    return g, provenance


def build(df, n_units, unit_idx):
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sa = pm.HalfNormal("sigma_a", 0.5)
        sb = pm.HalfNormal("sigma_b", 0.1)
        a = pm.Deterministic("a", mu_a + sa * pm.Normal("z_a", 0, 1, shape=n_units))
        b = pm.Deterministic("b", mu_b + sb * pm.Normal("z_b", 0, 1, shape=n_units))
        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(a[unit_idx] + b[unit_idx] * df.t.values
                           + np.log(df.births.values)),
            alpha=alpha,
            observed=df.avoidable.values,
        )
    return model


def unpooled(df):
    """Quasi-Poisson per unit, t on n-2 residual df: the study's standard everywhere."""
    n_fit = n_cls = n_rising = 0
    for _, g in df.groupby("unit"):
        if g.avoidable.sum() == 0 or g.births.sum() == 0:
            continue
        try:
            f = sm.GLM(g.avoidable, sm.add_constant(g[["t"]]),
                       family=sm.families.Poisson(), offset=np.log(g.births)).fit(scale="X2")
            b, se = float(f.params["t"]), float(f.bse["t"])
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
            continue
        n_fit += 1
        if abs(b) > stats.t.ppf(0.975, len(g) - 2) * se:
            n_cls += 1
            n_rising += b > 0
    return {"units_fitted": n_fit, "classified_unpooled": n_cls,
            "pct_unpooled": round(100 * n_cls / n_fit, 1), "rising_unpooled": n_rising}


def main():
    RES.joinpath("tables").mkdir(parents=True, exist_ok=True)
    df, provenance = panel()
    units = np.sort(df.unit.unique())
    idx = df.unit.map({u: i for i, u in enumerate(units)}).values

    cached = OUT / "idata_health_region.nc"
    if cached.exists():
        idata = az.from_netcdf(cached)
    else:
        with build(df, len(units), idx):
            idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
                              target_accept=TARGET_ACCEPT, progressbar=False)
        try:
            idata.to_netcdf(cached, engine="h5netcdf")
        except Exception:  # noqa: BLE001
            with open(OUT / "idata_health_region.pkl", "wb") as h:
                pickle.dump(idata, h)

    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo, hi = np.quantile(b, 0.025, axis=1), np.quantile(b, 0.975, axis=1)
    sd = b.std(axis=1)
    cls = (lo > 0) | (hi < 0)
    deaths = df.groupby("unit").avoidable.sum().loc[units].values

    pd.DataFrame({"regsaud_id": units, "deaths": deaths, "b_mean": b.mean(axis=1),
                  "b_sd": sd, "b_lo95": lo, "b_hi95": hi, "classified_95": cls}
                 ).to_csv(OUT / "slopes_health_region.csv", index=False)

    rhat = az.rhat(idata, var_names=["mu_b", "b"])
    result = {
        "level": "health region",
        "units": len(units),
        "median_deaths_per_unit": int(np.median(deaths)),
        "classified": int(cls.sum()),
        "pct_classified": round(100 * float(cls.mean()), 1),
        "rising_hierarchical": int((lo > 0).sum()),
        "median_posterior_sd": round(float(np.median(sd)), 4),
        "median_mde": round(float(np.median(1.96 * sd)), 4),
        "divergences": int(idata.sample_stats.diverging.sum()),
        "worst_rhat": round(max(float(rhat[v].max()) for v in rhat.data_vars), 4),
        **unpooled(df),
        "crosswalk_provenance": provenance,
    }

    # Read the immediate-region rung back so the comparison is against the committed
    # number rather than one retyped from the manuscript.
    ladder = pd.read_csv(RES / "tables" / "aggregation_ladder.csv").set_index("level")
    result["comparison_immediate_region"] = {
        "units": int(ladder.loc["immediate region", "units"]),
        "median_deaths_per_unit": int(ladder.loc["immediate region", "median_deaths_per_unit"]),
        "classified": int(ladder.loc["immediate region", "classified"]),
        "pct_classified": float(ladder.loc["immediate region", "pct_classified"]),
        "median_posterior_sd": float(ladder.loc["immediate region", "median_posterior_sd"]),
        "classified_unpooled": int(ladder.loc["immediate region", "classified_unpooled"]),
        "pct_unpooled": float(ladder.loc["immediate region", "pct_unpooled"]),
    }

    (RES / "HEALTH_REGION.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, dict)}]).to_csv(
        RES / "tables" / "health_region.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
