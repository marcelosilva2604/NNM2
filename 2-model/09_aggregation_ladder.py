"""Resolvability as a function of the unit of analysis, measured as a curve.

The two-point contrast between state and immediate region invites the objection that
the two levels were chosen because they make the point. It also confounds the unit with
the depth of the hierarchy, because the state model carried one level fewer.

Here four nested partitions of the same country and the same decade are fitted under one
specification: municipality, IBGE immediate region, state, and macro-region. Every rung
uses the same likelihood, the same estimand, the same two-level hierarchy (national plus
unit) and the same 95% criterion, so only the partition changes. The macro-region rung
is derived from the state code.

Municipalities are fitted unpooled as well, because at a median of 7 avoidable deaths
per municipality over the whole decade the hierarchical answer is almost entirely prior.

Run:
    .venv/bin/python 2-model/09_aggregation_ladder.py
"""

import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "2-model"
RES = ROOT / "3-results"

SEED = 20260803
DRAWS = 1500
TUNE = 2500
CHAINS = 4
TARGET_ACCEPT = 0.95

MACRO = {
    "AC": "North", "AM": "North", "AP": "North", "PA": "North", "RO": "North",
    "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast",
    "SE": "Northeast",
    "DF": "Centre-West", "GO": "Centre-West", "MS": "Centre-West", "MT": "Centre-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}


def panels():
    """Build the four nested partitions from the municipality panel."""
    muni = pd.read_csv(PROC / "panel_muni_year.csv")
    muni = muni.dropna(subset=["rgi_id"]).copy()
    muni["macro"] = muni.UF.map(MACRO)

    out = {}
    for name, key in [
        ("municipality", "CODMUNRES"),
        ("immediate region", "rgi_id"),
        ("state", "UF"),
        ("macro-region", "macro"),
    ]:
        g = muni.groupby([key, "year"], as_index=False)[["avoidable", "births"]].sum()
        g = g.rename(columns={key: "unit"})
        g = g[g.births > 0].copy()
        g["t"] = g.year - muni.year.mean()
        out[name] = g
    return out


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


def run_level(name, df):
    units = np.sort(df.unit.unique())
    idx = df.unit.map({u: i for i, u in enumerate(units)}).values

    tag = name.replace(" ", "_").replace("-", "_")
    cached = OUT / f"idata_ladder_{tag}.nc"
    if cached.exists():
        idata = az.from_netcdf(cached)
    else:
        with build(df, len(units), idx):
            idata = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
                target_accept=TARGET_ACCEPT, progressbar=False,
            )
        try:
            idata.to_netcdf(cached, engine="h5netcdf")
        except Exception:  # noqa: BLE001
            with open(OUT / f"idata_ladder_{tag}.pkl", "wb") as h:
                pickle.dump(idata, h)

    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo, hi = np.quantile(b, 0.025, axis=1), np.quantile(b, 0.975, axis=1)
    sd = b.std(axis=1)
    cls = (lo > 0) | (hi < 0)

    deaths = df.groupby("unit").avoidable.sum().loc[units].values
    pd.DataFrame(
        {"unit": units, "deaths": deaths, "b_mean": b.mean(axis=1), "b_sd": sd,
         "b_lo95": lo, "b_hi95": hi, "classified_95": cls}
    ).to_csv(OUT / f"slopes_ladder_{tag}.csv", index=False)

    rhat = az.rhat(idata, var_names=["mu_b", "b"])
    worst = max(float(rhat[v].max()) for v in rhat.data_vars)

    return {
        "level": name,
        "units": len(units),
        "median_deaths_per_unit": int(np.median(deaths)),
        "classified": int(cls.sum()),
        "pct_classified": round(100 * float(cls.mean()), 1),
        "median_posterior_sd": round(float(np.median(sd)), 4),
        "median_mde": round(float(np.median(1.96 * sd)), 4),
        "divergences": int(idata.sample_stats.diverging.sum()),
        "worst_rhat": round(worst, 4),
    }


def main():
    RES.joinpath("tables").mkdir(parents=True, exist_ok=True)
    data = panels()

    rows = []
    for name in ("municipality", "immediate region", "state", "macro-region"):
        rows.append(run_level(name, data[name]))
        print(rows[-1])

    table = pd.DataFrame(rows).merge(unpooled_ladder(), on="level")
    table.to_csv(RES / "tables" / "aggregation_ladder.csv", index=False)

    text = "\n".join(
        ["Aggregation ladder", "=" * 70, "", table.to_string(index=False)]
    )
    (OUT / "09_aggregation_ladder.log").write_text(text + "\n")
    print()
    print(text)


def unpooled_ladder():
    """Repeat the ladder without any hierarchy.

    At the coarsest rung the hierarchical model has five units, so its between-unit
    variance is unidentified (sigma_b 0.0128, 95% interval 0.0005 to 0.047) and every
    unit is shrunk to the national mean. The pooled rung is therefore uninformative
    there, and the unpooled fit is reported alongside so the ladder can be read at all.
    """
    import statsmodels.api as sm
    from scipy import stats

    rows = []
    for name, df in panels().items():
        n_cls = 0
        n_fit = 0
        for unit, g in df.groupby("unit"):
            if g.avoidable.sum() == 0 or g.births.sum() == 0:
                continue
            try:
                f = sm.GLM(
                    g.avoidable,
                    sm.add_constant(g[["t"]]),
                    family=sm.families.Poisson(),
                    offset=np.log(g.births),
                ).fit(scale="X2")
                b, se = float(f.params["t"]), float(f.bse["t"])
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
                continue
            n_fit += 1
            # t on n-2 residual df, not the normal (see 11_supporting_checks).
            n_cls += abs(b) > stats.t.ppf(0.975, len(g) - 2) * se
        rows.append(
            {
                "level": name,
                "units_fitted": n_fit,
                "classified_unpooled": n_cls,
                # Denominator is units that could be fitted unpooled. A unit with no
                # deaths has no unpooled slope, so this is smaller than the number of
                # units in the partition and the two columns condition on different
                # populations. Both denominators are reported.
                "pct_unpooled": round(100 * n_cls / n_fit, 1),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
