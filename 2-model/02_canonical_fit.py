"""The single canonical fit for Paper 2.

Every number in the manuscript must derive from the InferenceData objects this script
writes. Nothing is refitted downstream. The archived exploration refitted the model in
each of seven scripts under three different credible-interval conventions, which is why
the same count appeared as 85, 87, 88 and 95.

Two models, both on IBGE immediate regions (complete panel, zero-death unit-years kept):

    rate    NegativeBinomial count of avoidable neonatal deaths with a log-births offset.
            This is the primary estimand: deaths per birth, not composition.
    share   BetaBinomial count of avoidable deaths out of the four action groups.
            Secondary. Beta-binomial rather than binomial because the archived binomial
            ignored measured overdispersion.

Both carry a three-level slope (national + state + region) so the between-state share of
slope variance is a model parameter with a posterior, not an ANOVA on posterior means.

Convention: 95% equal-tailed intervals everywhere. No other width is used anywhere.

Run:
    .venv/bin/python 2-model/02_canonical_fit.py
"""

import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "panel_region_year.csv"
OUT = ROOT / "2-model"

HDI_PROB = 0.95
SEED = 20260803
# The first run at 1000/1000 left r-hat above 1.01 and bulk ESS under 100 for some
# parameters; the state-level scale is weakly identified by 27 states and needs the
# longer warm-up and the tighter step size.
DRAWS = 2000
TUNE = 3000
CHAINS = 4
TARGET_ACCEPT = 0.99


def load_panel():
    """Load the region-year panel and index regions and states for the hierarchy."""
    df = pd.read_csv(PANEL).sort_values(["rgi_id", "year"]).reset_index(drop=True)

    # Centre time on the midpoint so the intercept is the level in the middle of the
    # decade rather than an extrapolation to year zero.
    df["t"] = df["year"] - df["year"].mean()

    regions = np.sort(df.rgi_id.unique())
    df["region_idx"] = df.rgi_id.map({r: i for i, r in enumerate(regions)})

    region_state = (
        df[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    )
    states = np.sort(region_state.unique())
    state_of_region = region_state.map({s: i for s, i in zip(states, range(len(states)))}).values

    return df, regions, states, state_of_region


def build_model(df, n_regions, n_states, state_of_region, kind):
    """Assemble the rate or share model with a three-level intercept and slope."""
    coords = {"obs": np.arange(len(df))}

    with pm.Model(coords=coords) as model:
        t = pm.Data("t", df.t.values)
        r_idx = pm.Data("region_idx", df.region_idx.values)

        # National level and drift.
        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)

        # State and region deviations, non-centred.
        sigma_a_state = pm.HalfNormal("sigma_a_state", 0.5)
        sigma_a_region = pm.HalfNormal("sigma_a_region", 0.5)
        sigma_b_state = pm.HalfNormal("sigma_b_state", 0.1)
        sigma_b_region = pm.HalfNormal("sigma_b_region", 0.1)

        z_a_state = pm.Normal("z_a_state", 0.0, 1.0, shape=n_states)
        z_a_region = pm.Normal("z_a_region", 0.0, 1.0, shape=n_regions)
        z_b_state = pm.Normal("z_b_state", 0.0, 1.0, shape=n_states)
        z_b_region = pm.Normal("z_b_region", 0.0, 1.0, shape=n_regions)

        a_state = pm.Deterministic("a_state", sigma_a_state * z_a_state)
        b_state = pm.Deterministic("b_state", sigma_b_state * z_b_state)

        a = pm.Deterministic(
            "a", mu_a + a_state[state_of_region] + sigma_a_region * z_a_region
        )
        b = pm.Deterministic(
            "b", mu_b + b_state[state_of_region] + sigma_b_region * z_b_region
        )

        # The quantity the state-versus-region argument turns on, as a parameter.
        pm.Deterministic(
            "between_state_share",
            sigma_b_state**2 / (sigma_b_state**2 + sigma_b_region**2),
        )

        eta = a[r_idx] + b[r_idx] * t

        if kind == "rate":
            offset = pm.Data("log_births", np.log(df.births.values))
            alpha = pm.Exponential("alpha", 1.0)
            pm.NegativeBinomial(
                "y",
                mu=pm.math.exp(eta + offset),
                alpha=alpha,
                observed=df.avoidable.values,
            )
        elif kind == "share":
            n = pm.Data("n_trials", df.four_group.values)
            kappa = pm.Exponential("kappa", 1.0 / 100.0)
            # Kept out of the trace: one value per observation per draw would dominate
            # the stored posterior and slow every diagnostic that scans it.
            p = pm.math.sigmoid(eta)
            pm.BetaBinomial(
                "y",
                alpha=p * kappa,
                beta=(1.0 - p) * kappa,
                n=n,
                observed=df.avoidable.values,
            )
        else:
            raise ValueError(f"unknown model kind: {kind}")

    return model


def save_idata(idata, kind):
    """Persist the posterior, falling back to pickle rather than losing the sample.

    Sampling costs minutes; a missing serialisation backend must never discard it.
    """
    target = OUT / f"idata_{kind}.nc"
    try:
        idata.to_netcdf(target, engine="h5netcdf")
        return target
    except Exception as exc:  # noqa: BLE001 - any backend failure must fall through
        fallback = OUT / f"idata_{kind}.pkl"
        with open(fallback, "wb") as handle:
            pickle.dump(idata, handle)
        print(f"[{kind}] netCDF write failed ({exc}); wrote {fallback.name} instead")
        return fallback


def diagnostics(idata, kind):
    """Summarise convergence and the headline parameters for the run log."""
    keep = [
        "mu_a",
        "mu_b",
        "sigma_a_state",
        "sigma_a_region",
        "sigma_b_state",
        "sigma_b_region",
        "between_state_share",
    ]
    keep += ["alpha"] if kind == "rate" else ["kappa"]

    summary = az.summary(idata, var_names=keep, ci_prob=HDI_PROB)
    divergences = int(idata.sample_stats.diverging.sum())

    # Diagnose the parameters that carry the argument, including the per-region slopes.
    checked = keep + ["a", "b"]
    rhat = az.rhat(idata, var_names=checked)
    worst_rhat = float(max(float(rhat[v].max()) for v in rhat.data_vars))
    ess = az.ess(idata, var_names=checked)
    worst_ess = float(min(float(ess[v].min()) for v in ess.data_vars))

    lines = [
        f"[{kind}] divergences        : {divergences}",
        f"[{kind}] worst r-hat        : {worst_rhat:.4f}",
        f"[{kind}] lowest bulk ESS    : {worst_ess:.0f}",
        "",
        summary.to_string(),
    ]
    return "\n".join(lines)


def classification(idata, regions, kind):
    """Per-region slope posteriors, reported as precision rather than as a test.

    Reports the 95% criterion, the probability of direction, and the minimum detectable
    slope, so downstream scripts never have to re-derive any of them.
    """
    b = idata.posterior["b"].stack(sample=("chain", "draw")).values  # (region, sample)

    lo = np.quantile(b, 0.025, axis=1)
    hi = np.quantile(b, 0.975, axis=1)
    mean = b.mean(axis=1)
    sd = b.std(axis=1)
    p_dir = np.maximum((b > 0).mean(axis=1), (b < 0).mean(axis=1))

    out = pd.DataFrame(
        {
            "rgi_id": regions,
            "b_mean": mean,
            "b_sd": sd,
            "b_lo95": lo,
            "b_hi95": hi,
            "classified_95": (lo > 0) | (hi < 0),
            "prob_direction": p_dir,
            "mde": 1.96 * sd,  # minimum detectable slope at this unit's precision
        }
    )
    out.to_csv(OUT / f"slopes_{kind}.csv", index=False)

    n = len(out)
    lines = [
        f"[{kind}] regions                    : {n}",
        f"[{kind}] classified at 95%          : {out.classified_95.sum()}"
        f" ({out.classified_95.mean():.1%})",
        f"[{kind}] prob_direction >= 0.95     : {(p_dir >= 0.95).sum()}"
        f" ({(p_dir >= 0.95).mean():.1%})",
        f"[{kind}] median minimum detectable  : {np.median(out.mde):.4f} per year",
        f"[{kind}] naive sign error rate      : {(1 - p_dir).mean():.1%}",
    ]
    return "\n".join(lines)


def run(kind, df, regions, states, state_of_region, log):
    """Fit one model, save its InferenceData, and append its diagnostics to the log."""
    log.append(f"\n{'=' * 60}\nMODEL: {kind}\n{'=' * 60}")

    cached = OUT / f"idata_{kind}.nc"
    if cached.exists():
        # Resumable: sampling is the expensive step, so a completed fit is never redone.
        log.append(f"[{kind}] loaded cached posterior from {cached.name}")
        idata = az.from_netcdf(cached)
    else:
        model = build_model(df, len(regions), len(states), state_of_region, kind)
        with model:
            idata = pm.sample(
                draws=DRAWS,
                tune=TUNE,
                chains=CHAINS,
                random_seed=SEED,
                target_accept=TARGET_ACCEPT,
                progressbar=False,
            )
            # Posterior predictive draws are generated in 08_ppc_and_dispersion.py, not
            # here. An earlier version of this block extended a temporary thinned copy of
            # the InferenceData and saved the original, so no predictive group was ever
            # written; rather than repair a duplicate it is left to the script that
            # actually consumes it.
        save_idata(idata, kind)

    log.append(diagnostics(idata, kind))
    log.append("")
    log.append(classification(idata, regions, kind))
    return idata


def main():
    df, regions, states, state_of_region = load_panel()

    log = [
        "Paper 2 canonical fit",
        "=" * 60,
        f"panel rows            : {len(df):,}",
        f"immediate regions     : {len(regions):,}",
        f"states                : {len(states):,}",
        f"years                 : {df.year.min()}-{df.year.max()}",
        f"deaths (avoidable)    : {df.avoidable.sum():,}",
        f"deaths (four groups)  : {df.four_group.sum():,}",
        f"births                : {df.births.sum():,}",
        f"interval convention   : {HDI_PROB:.0%}",
        f"seed                  : {SEED}",
    ]

    for kind in ("rate", "share"):
        run(kind, df, regions, states, state_of_region, log)

    text = "\n".join(log)
    (OUT / "02_canonical_fit.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
