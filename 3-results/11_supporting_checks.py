"""The supporting analyses the methods review required, none of which need MCMC.

Six checks, each answering an objection a reviewer is likely to raise:

    coding      Did cause-of-death coding quality drift over the decade, and does that
                drift track the estimated slopes?
    allcause    How does resolvability look on the indicator the world actually reports,
                all-cause neonatal mortality, rather than on the avoidable subset?
    shock       How much of the residual year-to-year variation is a common national
                shock? If it is large, a shared temporal component is missing from the
                model; if small, that objection is answered in advance.
    linearity   In how many units does the unit's own data reject a straight line, and
                is that concentrated where the study can resolve anything?
    winner      By how much are the slopes of resolved units inflated relative to the
                rest, that is, the winner's curse.
    hetero      Is between-state dispersion in transition speed real when tested outside
                the hierarchical model, where shrinkage cannot create or hide it?

Run:
    .venv/bin/python 3-results/11_supporting_checks.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = ROOT / "2-model"
OUT = ROOT / "3-results"


def unpooled_slope(frame, outcome):
    """Quasi-Poisson log-linear slope from a unit's own series, with its own dispersion.

    Used wherever a claim must not depend on the hierarchical prior.
    """
    x = sm.add_constant(frame[["t"]])
    fit = sm.GLM(
        frame[outcome], x, family=sm.families.Poisson(), offset=np.log(frame.births)
    ).fit(scale="X2")
    return float(fit.params["t"]), float(fit.bse["t"])


def by_unit(panel, key, outcome, min_deaths=0):
    """Fit every unit separately and return slopes, standard errors and exposure."""
    rows = []
    for unit, g in panel.groupby(key):
        if g[outcome].sum() <= min_deaths or g.births.sum() == 0:
            continue
        try:
            b, se = unpooled_slope(g, outcome)
        except Exception:  # noqa: BLE001 - a unit whose GLM will not fit is reported as such
            continue
        if not np.isfinite(b) or not np.isfinite(se) or se == 0:
            continue
        rows.append({key: unit, "b": b, "se": se, "deaths": int(g[outcome].sum())})
    out = pd.DataFrame(rows)
    # A quasi-Poisson t ratio on n-2 residual degrees of freedom is referred to t, not
    # to the normal. With 11 annual points the normal critical value runs at a 7% to 8%
    # false positive rate instead of 5%.
    crit = stats.t.ppf(0.975, len(panel.year.unique()) - 2)
    out["classified"] = out.b.abs() > crit * out.se
    return out


def check_coding(region):
    """Ill-defined share over time, and whether its regional trend tracks the slopes."""
    by_year = region.groupby("year").apply(
        lambda g: g.illdef.sum() / g.deaths_total.sum(), include_groups=False
    )
    trends = []
    for rgi, g in region.groupby("rgi_id"):
        if g.deaths_total.sum() < 20:
            continue
        share = (g.illdef / g.deaths_total.replace(0, np.nan)).values
        ok = np.isfinite(share)
        if ok.sum() < 5:
            continue
        trends.append(
            {"rgi_id": rgi, "illdef_trend": np.polyfit(g.year.values[ok], share[ok], 1)[0]}
        )
    trends = pd.DataFrame(trends)
    slopes = pd.read_csv(MODEL / "slopes_rate.csv")[["rgi_id", "b_mean"]]
    merged = trends.merge(slopes, on="rgi_id")
    rho = stats.spearmanr(merged.illdef_trend, merged.b_mean)

    # Percentile bootstrap over regions for the correlation, 10,000 resamples, fixed seed.
    # The journal asks for an interval rather than a p-value; the Fisher transformation is
    # only approximate for Spearman, so the interval is resampled instead.
    rng = np.random.default_rng(20260904)
    x, y = merged.illdef_trend.values, merged.b_mean.values
    n = len(x)
    boot = np.empty(10_000)
    for i in range(boot.size):
        idx = rng.integers(0, n, n)
        boot[i] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "illdef_share_first": round(float(by_year.iloc[0]), 4),
        "illdef_share_last": round(float(by_year.iloc[-1]), 4),
        "illdef_share_min": round(float(by_year.min()), 4),
        "illdef_share_max": round(float(by_year.max()), 4),
        "n_regions_tested": int(len(merged)),
        "spearman_illdef_trend_vs_slope": round(float(rho.statistic), 3),
        "spearman_ci_lo": round(float(lo), 3),
        "spearman_ci_hi": round(float(hi), 3),
        "spearman_bootstrap": {"resamples": int(boot.size), "seed": 20260904, "method": "percentile"},
        "spearman_p": round(float(rho.pvalue), 4),
    }


def check_allcause(region):
    """Resolvability on three outcome definitions, all unpooled so the prior plays no part."""
    out = {}
    for name, col in [
        ("avoidable", "avoidable"),
        ("four_group", "four_group"),
        ("all_neonatal", "deaths_total"),
    ]:
        fits = by_unit(region, "rgi_id", col)
        out[name] = {
            "units_fitted": int(len(fits)),
            "classified": int(fits.classified.sum()),
            "pct": round(100 * float(fits.classified.mean()), 1),
            "median_deaths_per_unit": int(fits.deaths.median()),
        }
    return out


def check_shock(region, min_deaths=200):
    """Share of residual variance that is common to all regions in a given year.

    A large common component would mean the model is missing a shared temporal effect.
    """
    big = region.groupby("rgi_id").avoidable.sum()
    keep = big[big >= min_deaths].index
    sub = region[region.rgi_id.isin(keep)].copy()

    resid = {}
    for rgi, g in sub.groupby("rgi_id"):
        g = g.sort_values("year")
        rate = np.log((g.avoidable.values + 0.5) / g.births.values)
        fitted = np.polyval(np.polyfit(g.year.values, rate, 1), g.year.values)
        resid[rgi] = pd.Series(rate - fitted, index=g.year.values)
    matrix = pd.DataFrame(resid)  # years x regions

    year_mean = matrix.mean(axis=1)
    common = float(year_mean.var(ddof=0))
    total = float(matrix.values.var(ddof=0))
    return {
        "regions_used": int(matrix.shape[1]),
        "min_deaths": min_deaths,
        "common_variance_share": round(common / total, 4),
    }


def check_linearity(region):
    """How often a unit's own series rejects the straight line, by exposure quintile."""
    rows = []
    for rgi, g in region.groupby("rgi_id"):
        if g.avoidable.sum() < 10:
            continue
        g = g.sort_values("year")
        x = sm.add_constant(g[["t"]])
        lin = sm.GLM(
            g.avoidable, x, family=sm.families.Poisson(), offset=np.log(g.births)
        ).fit()
        p_gof = 1 - stats.chi2.cdf(float(lin.pearson_chi2), lin.df_resid)

        quad = g[["t"]].copy()
        quad["t2"] = quad.t**2
        q = sm.GLM(
            g.avoidable,
            sm.add_constant(quad),
            family=sm.families.Poisson(),
            offset=np.log(g.births),
        ).fit()
        p_curv = float(q.pvalues["t2"])

        rows.append(
            {
                "rgi_id": rgi,
                "deaths": int(g.avoidable.sum()),
                "reject_linear": p_gof < 0.05,
                "curvature": p_curv < 0.05,
            }
        )
    out = pd.DataFrame(rows)
    out["quintile"] = pd.qcut(out.deaths, 5, labels=False) + 1
    per_q = (
        out.groupby("quintile")
        .agg(
            median_deaths=("deaths", "median"),
            reject_pct=("reject_linear", lambda s: round(100 * s.mean(), 1)),
            curvature_pct=("curvature", lambda s: round(100 * s.mean(), 1)),
        )
        .reset_index()
    )
    return {
        "units_tested": int(len(out)),
        "reject_linear_pct": round(100 * float(out.reject_linear.mean()), 1),
        "by_quintile": per_q.to_dict("records"),
    }


def check_winner():
    """Magnitude inflation among resolved units relative to unresolved ones."""
    s = pd.read_csv(MODEL / "slopes_rate.csv")
    res = s[s.classified_95].b_mean.abs().median()
    unres = s[~s.classified_95].b_mean.abs().median()
    return {
        "median_abs_slope_resolved": round(float(res), 4),
        "median_abs_slope_unresolved": round(float(unres), 4),
        "inflation_ratio": round(float(res / unres), 2),
    }


def check_heterogeneity(region):
    """Between-state dispersion tested outside the hierarchical model.

    Cochran's Q and the DerSimonian-Laird estimate of the between-unit SD use each
    state's own series and its own dispersion, so neither shrinkage nor the prior can
    manufacture or conceal heterogeneity.
    """
    state = region.groupby(["UF", "year"], as_index=False)[["avoidable", "births"]].sum()
    state["t"] = state.year - region.year.mean()
    fits = by_unit(state, "UF", "avoidable")

    w = 1 / fits.se**2
    pooled = float((w * fits.b).sum() / w.sum())
    q = float((w * (fits.b - pooled) ** 2).sum())
    df = len(fits) - 1
    p = float(1 - stats.chi2.cdf(q, df))
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c)

    return {
        "states": int(len(fits)),
        "pooled_slope": round(pooled, 5),
        "Q": round(q, 1),
        "df": df,
        "p_value": f"{p:.2e}",
        "tau": round(float(np.sqrt(tau2)), 5),
        "tau_pct_per_decade": round(100 * (np.exp(np.sqrt(tau2) * 10) - 1), 1),
        "I2_pct": round(100 * max(0.0, (q - df) / q), 1),
        "states_classified_unpooled": int(fits.classified.sum()),
    }


def main():
    region = pd.read_csv(PROC / "panel_region_year.csv")
    region["t"] = region.year - region.year.mean()

    results = {
        "coding_quality": check_coding(region),
        "outcome_definition": check_allcause(region),
        "common_national_shock": check_shock(region),
        "linearity": check_linearity(region),
        "winners_curse": check_winner(),
        "between_state_heterogeneity": check_heterogeneity(region),
    }

    (OUT / "SUPPORTING_CHECKS.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
