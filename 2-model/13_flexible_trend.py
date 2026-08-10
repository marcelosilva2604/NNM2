"""Does the indeterminacy depend on summarising each region by a straight line?

A unit's own series rejects the log-linear fit in 13.3% of regions overall and in 26.5%
of the highest exposure quintile, which is where determinations are actually made. The
paper cannot report that and then leave the linear summary unchallenged.

Two alternatives are fitted on the same likelihood, hierarchy and criterion as the
canonical model, so only the shape of the time trend changes:

    quadratic   a second random coefficient on t squared, per region
    rw          a second-order random walk in time, shared nationally, added to the
                per-region linear trend; this absorbs common curvature without giving
                every region its own free shape

The reported quantity is the same one throughout the paper: how many regions have a
credibly non-zero average rate of change over the decade. For the flexible fits that is
the endpoint contrast, the change from the first to the last year, divided by the span,
so it is comparable to the linear slope rather than to a coefficient that has no
counterpart in the simpler model.

Run:
    .venv/bin/python 2-model/13_flexible_trend.py
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
    years = np.sort(df.year.unique())
    df["year_idx"] = df.year.map({y: i for i, y in enumerate(years)})
    rs = df[["rgi_id", "UF"]].drop_duplicates().set_index("rgi_id").loc[regions, "UF"]
    states = np.sort(rs.unique())
    sor = rs.map({s: i for i, s in enumerate(states)}).values
    return df, regions, states, sor, years


def build(df, n_regions, n_states, sor, n_years, shape):
    with pm.Model() as model:
        r_idx = df.region_idx.values
        y_idx = df.year_idx.values
        t = df.t.values

        mu_a = pm.Normal("mu_a", 0.0, 5.0)
        mu_b = pm.Normal("mu_b", 0.0, 0.5)
        sa_s = pm.HalfNormal("sigma_a_state", 0.5)
        sa_r = pm.HalfNormal("sigma_a_region", 0.5)
        sb_s = pm.HalfNormal("sigma_b_state", 0.1)
        sb_r = pm.HalfNormal("sigma_b_region", 0.1)

        a = pm.Deterministic(
            "a",
            mu_a + (sa_s * pm.Normal("z_a_state", 0, 1, shape=n_states))[sor]
            + sa_r * pm.Normal("z_a_region", 0, 1, shape=n_regions),
        )
        b = pm.Deterministic(
            "b",
            mu_b + (sb_s * pm.Normal("z_b_state", 0, 1, shape=n_states))[sor]
            + sb_r * pm.Normal("z_b_region", 0, 1, shape=n_regions),
        )

        eta = a[r_idx] + b[r_idx] * t

        if shape == "quadratic":
            mu_c = pm.Normal("mu_c", 0.0, 0.1)
            sc = pm.HalfNormal("sigma_c", 0.05)
            c = pm.Deterministic(
                "c", mu_c + sc * pm.Normal("z_c", 0, 1, shape=n_regions)
            )
            eta = eta + c[r_idx] * (t**2)

        elif shape == "rw":
            # Second-order random walk on the national year effect, centred out so it
            # cannot absorb the level or the linear trend it is meant to complement.
            sigma_w = pm.HalfNormal("sigma_w", 0.05)
            steps = pm.Normal("w_steps", 0.0, 1.0, shape=n_years - 2)
            second = pm.math.concatenate([[0.0, 0.0], sigma_w * steps])
            w_raw = pm.math.cumsum(pm.math.cumsum(second))
            tt = np.arange(n_years) - (n_years - 1) / 2.0
            # Remove the constant and linear components by projection.
            w = w_raw - pm.math.mean(w_raw)
            w = w - (pm.math.sum(w * tt) / np.sum(tt**2)) * tt
            w = pm.Deterministic("w", w)
            eta = eta + w[y_idx]

        elif shape != "linear":
            raise ValueError(shape)

        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial(
            "y",
            mu=pm.math.exp(eta + np.log(df.births.values)),
            alpha=alpha,
            observed=df.avoidable.values,
        )
    return model


def average_rate(idata, shape, years):
    """Average annual rate of change per region, comparable across trend shapes.

    For the linear and random-walk fits the per-region trend is b itself, because the
    national year effect is common and has had its linear part projected out.

    For the quadratic fit the average derivative over a symmetrically centred window is
    also exactly b: the curvature term contributes c*(t_last + t_first), which is zero by
    construction. The arm therefore tests whether admitting curvature disturbs the linear
    summary, which it does through b's posterior, and NOT whether curvature is itself
    detectable. It should not be described as the latter.
    """
    post = idata.posterior
    b = post["b"].stack(sample=("chain", "draw")).values
    if shape != "quadratic":
        return b

    t_first = years.min() - years.mean()
    t_last = years.max() - years.mean()
    c = post["c"].stack(sample=("chain", "draw")).values
    span = t_last - t_first
    return b + c * (t_last + t_first)  # d(eta)/dt averaged over the decade


def main():
    df, regions, states, sor, years = load()
    rows, log = [], ["Flexible trend shapes", "=" * 60]

    for shape in ("linear", "quadratic", "rw"):
        cached = OUT / f"idata_shape_{shape}.nc"
        if cached.exists():
            idata = az.from_netcdf(cached)
        else:
            with build(df, len(regions), len(states), sor, len(years), shape):
                idata = pm.sample(
                    draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=SEED,
                    target_accept=TARGET_ACCEPT, progressbar=False,
                )
            try:
                idata.to_netcdf(cached, engine="h5netcdf")
            except Exception:  # noqa: BLE001
                with open(OUT / f"idata_shape_{shape}.pkl", "wb") as h:
                    pickle.dump(idata, h)

        rate = average_rate(idata, shape, years)
        lo, hi = np.quantile(rate, 0.025, axis=1), np.quantile(rate, 0.975, axis=1)
        cls = (lo > 0) | (hi < 0)
        sd = rate.std(axis=1)

        pd.DataFrame(
            {"rgi_id": regions, "rate_mean": rate.mean(axis=1), "rate_sd": sd,
             "lo95": lo, "hi95": hi, "classified_95": cls}
        ).to_csv(OUT / f"slopes_shape_{shape}.csv", index=False)

        checked = ["mu_b", "a", "b"]
        rhat = az.rhat(idata, var_names=checked)
        worst = max(float(rhat[v].max()) for v in rhat.data_vars)

        rows.append(
            {
                "trend": shape,
                "classified": int(cls.sum()),
                "pct": round(100 * float(cls.mean()), 1),
                "median_sd": round(float(np.median(sd)), 4),
                "alpha": round(float(idata.posterior["alpha"].mean()), 1),
                "divergences": int(idata.sample_stats.diverging.sum()),
                "worst_rhat": round(worst, 4),
            }
        )
        log.append(f"[{shape}] {rows[-1]}")
        print(rows[-1])

    table = pd.DataFrame(rows)
    table.to_csv(RES / "tables" / "trend_shape.csv", index=False)
    log.append("")
    log.append(table.to_string(index=False))
    (OUT / "13_flexible_trend.log").write_text("\n".join(log) + "\n")
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
