"""Posterior predictive checks, and a variance function that is measured rather than imposed.

Two defects in the canonical fit are addressed here.

First, the canonical script intended to store posterior predictive draws but never did:
it passed a thinned copy of the InferenceData to the sampler, extended that temporary
copy, and saved the original. No predictive group was ever written and nothing
downstream consumed one, so the omission went unnoticed. This script generates the
draws correctly and reports the checks.

Second, the canonical model gives every unit a single negative binomial dispersion,
which imposes an identical floor on the coefficient of variation regardless of unit
size. Any statement about how precision varies with exposure is then a property of that
assumption rather than a finding. Here the dispersion is allowed to depend on exposure,
log alpha_i = c0 + c1 * log(mean births), so the relationship is estimated.

Run:
    .venv/bin/python 2-model/08_ppc_and_dispersion.py
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
DRAWS = 2000
TUNE = 3000
CHAINS = 4
TARGET_ACCEPT = 0.99


def load():
    df = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"])
    df = df.reset_index(drop=True)
    df["t"] = df.year - df.year.mean()
    regions = np.sort(df.rgi_id.unique())
    df["region_idx"] = df.rgi_id.map({r: i for i, r in enumerate(regions)})
    rs = df[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    states = np.sort(rs.unique())
    sor = rs.map({s: i for i, s in enumerate(states)}).values

    # Exposure is centred on the log scale so c0 is the dispersion of a typical region.
    exposure = df.groupby("region_idx").births.mean().loc[np.arange(len(regions))].values
    log_exp = np.log(exposure)
    log_exp_c = log_exp - log_exp.mean()
    return df, regions, states, sor, log_exp_c


def build(df, n_regions, n_states, sor, log_exp_c, dispersion):
    with pm.Model() as model:
        t = df.t.values
        r_idx = df.region_idx.values

        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sa_s = pm.HalfNormal("sigma_a_state", 0.5)
        sa_r = pm.HalfNormal("sigma_a_region", 0.5)
        sb_s = pm.HalfNormal("sigma_b_state", 0.1)
        sb_r = pm.HalfNormal("sigma_b_region", 0.1)

        a = pm.Deterministic(
            "a",
            mu_a
            + (sa_s * pm.Normal("z_a_state", 0, 1, shape=n_states))[sor]
            + sa_r * pm.Normal("z_a_region", 0, 1, shape=n_regions),
        )
        b = pm.Deterministic(
            "b",
            mu_b
            + (sb_s * pm.Normal("z_b_state", 0, 1, shape=n_states))[sor]
            + sb_r * pm.Normal("z_b_region", 0, 1, shape=n_regions),
        )

        if dispersion == "global":
            alpha_obs = pm.Exponential("alpha", 1.0)
        elif dispersion == "exposure":
            c0 = pm.Normal("log_alpha0", np.log(50.0), 1.5)
            c1 = pm.Normal("log_alpha_slope", 0.0, 1.0)
            alpha_region = pm.Deterministic(
                "alpha_region", pm.math.exp(c0 + c1 * log_exp_c)
            )
            alpha_obs = alpha_region[r_idx]
        else:
            raise ValueError(dispersion)

        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(a[r_idx] + b[r_idx] * t + np.log(df.births.values)),
            alpha=alpha_obs,
            observed=df.avoidable.values,
        )
    return model


def fit(df, regions, states, sor, log_exp_c, dispersion):
    """Sample, then attach posterior predictive draws to the SAME object that is saved."""
    cached = OUT / f"idata_disp_{dispersion}.nc"
    if cached.exists():
        return az.from_netcdf(cached)

    model = build(df, len(regions), len(states), sor, log_exp_c, dispersion)
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=SEED,
            target_accept=TARGET_ACCEPT,
            progressbar=False,
        )
        # The predictive group is attached to idata itself. Thinning is applied by
        # assigning the returned group back, never by extending a temporary copy.
        ppc = pm.sample_posterior_predictive(
            idata, random_seed=SEED, progressbar=False, return_inferencedata=True
        )
    idata["posterior_predictive"] = ppc["posterior_predictive"]
    try:
        idata.to_netcdf(cached, engine="h5netcdf")
    except Exception:  # noqa: BLE001
        with open(OUT / f"idata_disp_{dispersion}.pkl", "wb") as h:
            pickle.dump(idata, h)
    return idata


def ppc_report(idata, df, label):
    """Pearson residual check by exposure quintile, and coverage of the predictive intervals."""
    y = df.avoidable.values
    rep = idata["posterior_predictive"]["y"]
    rep = rep.stack(sample=("chain", "draw")).values  # (obs, sample)

    mean = rep.mean(axis=1)
    var = rep.var(axis=1)
    pearson = (y - mean) / np.sqrt(np.maximum(var, 1e-9))

    lo = np.quantile(rep, 0.025, axis=1)
    hi = np.quantile(rep, 0.975, axis=1)
    covered = (y >= lo) & (y <= hi)

    # Divide by residual degrees of freedom, not by the number of observations. Each
    # unit spends two parameters, so a correctly specified model gives roughly 9/11 under
    # the wrong divisor and would be read as overdispersed.
    resid_df = len(df) - 2 * df.rgi_id.nunique()
    scale = len(df) / resid_df
    pooled = df.groupby("rgi_id").avoidable.transform("sum")
    q = pd.qcut(pooled, 5, labels=False) + 1
    tab = (
        pd.DataFrame({"q": q, "chi2": pearson**2, "cov": covered, "mu": mean})
        .groupby("q")
        .agg(
            mean_fitted=("mu", "mean"),
            chi2_over_df=("chi2", lambda v: v.mean() * scale),
            coverage=("cov", "mean"),
        )
        .round(3)
        .reset_index()
    )
    lines = [
        f"[{label}] posterior predictive check",
        f"  overall Pearson chi2/df : {float((pearson ** 2).sum() / resid_df):.3f}",
        f"  overall 95% coverage    : {float(covered.mean()):.3f}",
        "",
        tab.to_string(index=False),
    ]
    return "\n".join(lines), tab


def classify(idata, regions, label):
    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo, hi = np.quantile(b, 0.025, axis=1), np.quantile(b, 0.975, axis=1)
    sd = b.std(axis=1)
    cls = (lo > 0) | (hi < 0)
    pd.DataFrame(
        {"rgi_id": regions, "b_mean": b.mean(axis=1), "b_sd": sd,
         "b_lo95": lo, "b_hi95": hi, "classified_95": cls}
    ).to_csv(OUT / f"slopes_disp_{label}.csv", index=False)
    return cls, sd


def main():
    df, regions, states, sor, log_exp_c = load()
    log = ["Posterior predictive checks and exposure-dependent dispersion", "=" * 70]

    summary = {}
    for dispersion in ("global", "exposure"):
        idata = fit(df, regions, states, sor, log_exp_c, dispersion)
        text, _ = ppc_report(idata, df, dispersion)
        log.append("")
        log.append(text)

        cls, sd = classify(idata, regions, dispersion)
        summary[dispersion] = {
            "classified": int(cls.sum()),
            "pct": round(100 * float(cls.mean()), 1),
            "median_sd": round(float(np.median(sd)), 4),
            "sd_ratio_max_min": round(float(sd.max() / sd.min()), 2),
            "divergences": int(idata.sample_stats.diverging.sum()),
        }
        log.append("")
        log.append(f"  classified at 95%       : {summary[dispersion]['classified']}/510")
        log.append(f"  median posterior SD     : {summary[dispersion]['median_sd']}")
        log.append(f"  SD dynamic range        : {summary[dispersion]['sd_ratio_max_min']}x")

        if dispersion == "exposure":
            s = az.summary(
                idata, var_names=["log_alpha0", "log_alpha_slope"], ci_prob=0.95
            )
            log.append("")
            log.append("  dispersion as a function of exposure:")
            log.append(s.to_string())

    log.append("")
    log.append(pd.DataFrame(summary).T.to_string())
    text = "\n".join(log)
    (OUT / "08_ppc_and_dispersion.log").write_text(text + "\n")
    (RES / "tables" / "dispersion_comparison.csv").write_text(
        pd.DataFrame(summary).T.to_csv()
    )
    print(text)


if __name__ == "__main__":
    main()
