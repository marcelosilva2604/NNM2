"""Fit the same model at state level, so state and region can be compared on equal terms.

The paper's argument is that a country can look well monitored at the level where it
reports and be unreadable at the level where it acts. Testing that requires the state
answer to come from the same likelihood, the same estimand and the same criterion as the
region answer, differing only in the unit. This script provides the state half.

Unit: the 27 federative units. Likelihood: negative binomial on avoidable neonatal deaths
with a log-births offset, national intercept and slope plus a state deviation on each.
Criterion: 95% equal-tailed interval excluding zero, as everywhere else in this study.

Run:
    .venv/bin/python 2-model/07_state_model.py
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

SEED = 20260803
DRAWS = 2000
TUNE = 3000
CHAINS = 4
TARGET_ACCEPT = 0.99


def load():
    """Collapse the region panel to state-year."""
    df = pd.read_csv(PROC / "panel_region_year.csv")
    state = (
        df.groupby(["UF", "year"], as_index=False)[["avoidable", "births", "deaths_total"]]
        .sum()
        .sort_values(["UF", "year"])
        .reset_index(drop=True)
    )
    # Centre time exactly as the region model does, so the slopes are comparable.
    state["t"] = state["year"] - df["year"].mean()

    states = np.sort(state.UF.unique())
    state["state_idx"] = state.UF.map({s: i for i, s in enumerate(states)})
    return state, states


def build(df, n_states):
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)

        sigma_a = pm.HalfNormal("sigma_a", 0.5)
        sigma_b = pm.HalfNormal("sigma_b", 0.1)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_states)
        z_b = pm.Normal("z_b", 0.0, 1.0, shape=n_states)

        a = pm.Deterministic("a", mu_a + sigma_a * z_a)
        b = pm.Deterministic("b", mu_b + sigma_b * z_b)

        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(
                a[df.state_idx.values]
                + b[df.state_idx.values] * df.t.values
                + np.log(df.births.values)
            ),
            alpha=alpha,
            observed=df.avoidable.values,
        )
    return model


def main():
    df, states = load()

    cached = OUT / "idata_state.nc"
    if cached.exists():
        idata = az.from_netcdf(cached)
    else:
        with build(df, len(states)):
            idata = pm.sample(
                draws=DRAWS,
                tune=TUNE,
                chains=CHAINS,
                random_seed=SEED,
                target_accept=TARGET_ACCEPT,
                progressbar=False,
            )
        try:
            idata.to_netcdf(cached, engine="h5netcdf")
        except Exception:  # noqa: BLE001
            with open(OUT / "idata_state.pkl", "wb") as handle:
                pickle.dump(idata, handle)

    b = idata.posterior["b"].stack(sample=("chain", "draw")).values
    lo = np.quantile(b, 0.025, axis=1)
    hi = np.quantile(b, 0.975, axis=1)
    out = pd.DataFrame(
        {
            "UF": states,
            "b_mean": b.mean(axis=1),
            "b_sd": b.std(axis=1),
            "b_lo95": lo,
            "b_hi95": hi,
            "classified_95": (lo > 0) | (hi < 0),
        }
    )
    out.to_csv(OUT / "slopes_state.csv", index=False)

    rhat = az.rhat(idata, var_names=["mu_b", "a", "b"])
    worst = max(float(rhat[v].max()) for v in rhat.data_vars)
    ess = az.ess(idata, var_names=["mu_b", "a", "b"])
    lowest = min(float(ess[v].min()) for v in ess.data_vars)

    lines = [
        "State-level fit",
        "=" * 50,
        f"states                : {len(states)}",
        f"state-years           : {len(df)}",
        f"divergences           : {int(idata.sample_stats.diverging.sum())}",
        f"worst r-hat           : {worst:.4f}",
        f"lowest bulk ESS       : {lowest:.0f}",
        f"national drift mu_b   : {float(idata.posterior['mu_b'].mean()):+.4f} per year",
        "",
        f"classified at 95%     : {int(out.classified_95.sum())}/{len(states)}"
        f" ({out.classified_95.mean():.1%})",
        f"  of which falling    : {int((out.classified_95 & (out.b_mean < 0)).sum())}",
        f"  of which rising     : {int((out.classified_95 & (out.b_mean > 0)).sum())}",
        f"unresolved states     : "
        f"{', '.join(out.loc[~out.classified_95, 'UF'].tolist()) or 'none'}",
        "",
        out.round(4).to_string(index=False),
    ]
    text = "\n".join(lines)
    (OUT / "07_state_model.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
