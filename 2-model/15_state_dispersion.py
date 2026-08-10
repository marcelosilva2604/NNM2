"""Refit the state model with a dispersion that fits the large states.

The state model in `07_state_model.py` carries a single negative binomial dispersion for
all 27 states, which imposes a common floor on the coefficient of variation regardless of
size. At the fitted alpha of 59.5 that floor is about 13% of the mean, while the observed
residual variability around a log-linear fit is roughly 4% in Sao Paulo and Rio de
Janeiro. The model therefore injects about three times too much slope noise into the
largest states, which is why every state's posterior standard deviation sits in a narrow
band regardless of exposure.

This is the same single-dispersion pathology that `08_ppc_and_dispersion.py` diagnosed and
repaired at region level, and it was never applied here. It matters because a conclusion
depends on it: under the single-dispersion model no state's slope separates from the
national drift, whereas the same unpooled machinery this study uses to compute Cochran's
Q does separate several. Certifying heterogeneity with one instrument and refusing to read
its components with another is not defensible, so the two must be reconciled.

Two fits, identical apart from the variance function:

    global      one dispersion for all states (the published specification)
    exposure    log alpha_i = c0 + c1 * log(mean births), as at region level

Both report, for every state, the slope, the contrast against the national drift, and
whether that contrast excludes zero. The unpooled contrast is computed alongside, each
state against the rest of the country, referred to t on n-2 residual degrees of freedom.

Run:
    .venv/bin/python 2-model/15_state_dispersion.py
"""

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
OUT = ROOT / "2-model"
RES = ROOT / "3-results"

SEED = 20260803
DRAWS = 2000
TUNE = 3000
CHAINS = 4
TARGET_ACCEPT = 0.99


def load():
    d = pd.read_csv(PROC / "panel_region_year.csv")
    st = d.groupby(["UF", "year"], as_index=False)[["avoidable", "births"]].sum()
    st["t"] = st.year - d.year.mean()
    states = np.sort(st.UF.unique())
    st["idx"] = st.UF.map({s: i for i, s in enumerate(states)})
    exposure = st.groupby("idx").births.mean().loc[np.arange(len(states))].values
    log_exp_c = np.log(exposure) - np.log(exposure).mean()
    return st, states, log_exp_c


def build(df, n_states, log_exp_c, dispersion):
    with pm.Model() as model:
        idx = df.idx.values
        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sa = pm.HalfNormal("sigma_a", 0.5)
        sb = pm.HalfNormal("sigma_b", 0.1)
        a = pm.Deterministic("a", mu_a + sa * pm.Normal("z_a", 0, 1, shape=n_states))
        b = pm.Deterministic("b", mu_b + sb * pm.Normal("z_b", 0, 1, shape=n_states))

        if dispersion == "global":
            alpha = pm.Exponential("alpha", 1.0)
        else:
            c0 = pm.Normal("log_alpha0", np.log(60.0), 1.5)
            c1 = pm.Normal("log_alpha_slope", 0.0, 1.0)
            alpha = pm.Deterministic("alpha_state", pm.math.exp(c0 + c1 * log_exp_c))[idx]

        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(a[idx] + b[idx] * df.t.values + np.log(df.births.values)),
            alpha=alpha,
            observed=df.avoidable.values,
        )
    return model


def unpooled_contrasts(st, mean_year):
    """Each state against the rest of the country, using each fit's own dispersion."""
    def fit(x):
        f = sm.GLM(
            x.avoidable, sm.add_constant(x[["t"]]),
            family=sm.families.Poisson(), offset=np.log(x.births),
        ).fit(scale="X2")
        return float(f.params["t"]), float(f.bse["t"])

    rows = []
    for uf, x in st.groupby("UF"):
        b, se = fit(x)
        rest = st[st.UF != uf].groupby("year", as_index=False)[["avoidable", "births"]].sum()
        rest["t"] = rest.year - mean_year
        bn, sen = fit(rest)
        diff = b - bn
        sed = float(np.hypot(se, sen))
        t = diff / sed
        rows.append(
            {
                "UF": uf, "slope_own": b, "se_own": se, "slope_rest": bn,
                "contrast": diff, "t": t,
                "p": float(2 * (1 - stats.t.cdf(abs(t), 9))),
            }
        )
    out = pd.DataFrame(rows)
    out["departs_unpooled"] = out.p < 0.05
    return out


def main():
    st, states, log_exp_c = load()
    mean_year = pd.read_csv(PROC / "panel_region_year.csv").year.mean()

    unp = unpooled_contrasts(st, mean_year)
    log = ["State model: dispersion and attribution", "=" * 70, ""]

    results = {}
    for dispersion in ("global", "exposure"):
        cached = OUT / f"idata_state_{dispersion}.nc"
        if cached.exists():
            idata = az.from_netcdf(cached)
        else:
            with build(st, len(states), log_exp_c, dispersion):
                idata = pm.sample(
                    draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
                    target_accept=TARGET_ACCEPT, progressbar=False,
                )
            try:
                idata.to_netcdf(cached, engine="h5netcdf")
            except Exception:  # noqa: BLE001
                with open(OUT / f"idata_state_{dispersion}.pkl", "wb") as h:
                    pickle.dump(idata, h)

        post = idata.posterior
        b = post["b"].stack(sample=("chain", "draw")).values
        mu = post["mu_b"].stack(sample=("chain", "draw")).values
        delta = b - mu[None, :]

        lo, hi = np.quantile(b, 0.025, axis=1), np.quantile(b, 0.975, axis=1)
        dlo, dhi = np.quantile(delta, 0.025, axis=1), np.quantile(delta, 0.975, axis=1)

        frame = pd.DataFrame(
            {
                "UF": states,
                f"b_{dispersion}": b.mean(axis=1),
                f"sd_{dispersion}": b.std(axis=1),
                f"classified_{dispersion}": (lo > 0) | (hi < 0),
                f"departs_{dispersion}": (dlo > 0) | (dhi < 0),
            }
        )
        results[dispersion] = frame

        log.append(
            f"[{dispersion}] classified {int(frame[f'classified_{dispersion}'].sum())}/27 | "
            f"depart from national {int(frame[f'departs_{dispersion}'].sum())}/27 | "
            f"median posterior SD {np.median(b.std(axis=1)):.4f} | "
            f"SD range {b.std(axis=1).min():.4f}-{b.std(axis=1).max():.4f} | "
            f"divergences {int(idata.sample_stats.diverging.sum())}"
        )
        if dispersion == "exposure":
            s = post["log_alpha_slope"].stack(sample=("chain", "draw")).values
            log.append(
                f"           dispersion-on-exposure coefficient "
                f"{s.mean():.4f} ({np.quantile(s, 0.025):.4f} to {np.quantile(s, 0.975):.4f})"
            )

    merged = results["global"].merge(results["exposure"], on="UF").merge(unp, on="UF")
    merged.to_csv(RES / "tables" / "state_dispersion.csv", index=False)

    log += [
        "",
        f"unpooled: {int(unp.departs_unpooled.sum())}/27 states depart from the rest of "
        f"the country at p<0.05 (t on 9 df)",
        "",
        merged[
            ["UF", "b_global", "sd_global", "departs_global", "b_exposure",
             "sd_exposure", "departs_exposure", "slope_own", "t", "p", "departs_unpooled"]
        ].round(4).to_string(index=False),
    ]
    text = "\n".join(log)
    (OUT / "15_state_dispersion.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
