"""Where does unpooled power reach 80%? Arm A extended beyond the observed exposure range.

The committed design analysis (16_design_two_arm.py, arm A) evaluates power at the 510
immediate regions' own livebirth counts, so the largest exposure it ever sees is the top
quintile (median 79.1 all-cause neonatal deaths per region-year, power 13.8% against the
national change). The manuscript therefore could not say how many deaths per unit-year
a registry would need before a change the size of the national one becomes detectable.
This script answers that with two independent routes.

    Route 1, simulation. Arm A is re-run with every region's livebirths multiplied by a
    scale factor k in {1, 2, 4, ..., 256}; the per-region dispersion is scaled with it
    along the fitted exposure relation (log alpha_i = c0 + c1 log births_i, from
    08_ppc_and_dispersion.py), so the simulated series stay on the variance function the
    project's own predictive check prefers. Counts are drawn under three true slopes
    (national, -8.0% per decade; the median minimum detectable slope under pooling,
    -14.3% per decade; -20% per decade) and
    under the null, and each series is refitted exactly as in arm A: quasi-Poisson,
    unit-specific Pearson dispersion, t on n-2 = 9 residual degrees of freedom. The
    unpooled fit is implemented here as a vectorised IRLS and checked against
    statsmodels on a sample of series before any result is written.

    Route 2, analytic. For a unit with D expected deaths per year the slope's sampling
    variance follows from the Poisson information matrix with a negative binomial
    variance (sandwich form), with alpha taken from the same fitted exposure relation,
    and power from the non-central t on 9 degrees of freedom. No simulation enters.

Both routes are reported on one axis, all-cause neonatal deaths per unit-year, and the
exposure at which power crosses 80% (and 50%) is read off each. The simulated threshold
is log-linear interpolation of the binned power curve; the analytic one is a root find.

Writes (all-cause results directory):
    tables/power_threshold_armA.csv    power, type S, type M by quintile, scale and slope
    tables/power_threshold_curve.csv   binned power curve, simulated and analytic
    POWER_THRESHOLD.json               the thresholds and the inputs they rest on
    figures/power_threshold.png/.pdf   the curve

Run:
    NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/23_power_threshold.py
"""

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, LABEL, model_dir, results_dir  # noqa: E402
PROC = ROOT / "data" / "processed"
MODEL = model_dir(ROOT)
OUT = results_dir(ROOT)

RNG = np.random.default_rng(20261011)
SIMS = 20                                       # per region, per scenario (as arm A)
SCALES = (1, 2, 4, 8, 16, 32, 64, 128, 256)     # multipliers on observed livebirths
N_YEARS = 11
T_CRIT = stats.t.ppf(0.975, N_YEARS - 2)
TARGETS = (0.5, 0.8)
BIN_EDGES = 2.0 ** np.arange(2, 15.5, 0.5)     # 4 to ~16,000 deaths per unit-year


# --------------------------------------------------------------------------- inputs
def setup():
    panel = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"])
    panel = panel.reset_index(drop=True)
    panel["t"] = panel.year - panel.year.mean()
    regions = np.sort(panel.rgi_id.unique())
    panel["region_idx"] = panel.rgi_id.map({r: i for i, r in enumerate(regions)})

    canonical = az.from_netcdf(MODEL / "idata_rate.nc")
    mu_b = float(canonical.posterior["mu_b"].mean())

    disp = az.from_netcdf(MODEL / "idata_disp_exposure.nc")
    alpha = disp.posterior["alpha_region"].mean(dim=("chain", "draw")).values
    c0 = float(disp.posterior["log_alpha0"].mean())
    c1 = float(disp.posterior["log_alpha_slope"].mean())
    # Centring used by 08_ppc_and_dispersion.py: mean over regions of log(mean births).
    mean_births = panel.groupby("region_idx").births.mean().loc[np.arange(len(regions))]
    log_exp_centre = float(np.log(mean_births.values).mean())

    level = (
        panel.groupby("rgi_id")
        .apply(lambda g: np.log(g[OUTCOME].sum() / g.births.sum()), include_groups=False)
        .loc[regions]
        .values
    )
    slopes = pd.read_csv(MODEL / "slopes_rate.csv")
    median_mde = float(slopes.mde.median())

    pooled = panel.groupby("rgi_id")[OUTCOME].sum().loc[regions].values
    quint = pd.qcut(pooled, 5, labels=False) + 1
    national_rate = panel[OUTCOME].sum() / panel.births.sum()
    return dict(panel=panel, regions=regions, mu_b=mu_b, alpha=alpha, c0=c0, c1=c1,
                log_exp_centre=log_exp_centre, level=level, median_mde=median_mde,
                pooled=pooled, quint=quint, national_rate=float(national_rate))


# --------------------------------------------------------------------------- unpooled fit
def fit_quasipoisson(Y, O, t):
    """Vectorised quasi-Poisson log-linear fit of y ~ 1 + t with offset, per row.

    Returns slope, its standard error (Pearson scale on n-2 df) and a validity flag.
    Equivalent to statsmodels GLM(Poisson, offset).fit(scale="X2"); checked in main().
    """
    n_series, n = Y.shape
    ok = Y.sum(axis=1) > 0
    a = np.where(ok, np.log(np.maximum(Y.sum(axis=1), 1) / np.exp(O).sum(axis=1)), 0.0)
    b = np.zeros(n_series)
    converged = np.zeros(n_series, dtype=bool)
    for _ in range(100):
        eta = O + a[:, None] + b[:, None] * t[None, :]
        mu = np.exp(eta)
        z = (eta - O) + (Y - mu) / mu
        Sw, St, Stt = mu.sum(1), (mu * t).sum(1), (mu * t * t).sum(1)
        Sz, Stz = (mu * z).sum(1), (mu * t * z).sum(1)
        det = Sw * Stt - St * St
        a_new = (Stt * Sz - St * Stz) / det
        b_new = (Sw * Stz - St * Sz) / det
        step = np.maximum(np.abs(a_new - a), np.abs(b_new - b))
        a, b = a_new, b_new
        converged = step < 1e-10
        if converged[ok].all():
            break
    eta = O + a[:, None] + b[:, None] * t[None, :]
    mu = np.exp(eta)
    Sw, St, Stt = mu.sum(1), (mu * t).sum(1), (mu * t * t).sum(1)
    det = Sw * Stt - St * St
    chi2 = ((Y - mu) ** 2 / mu).sum(1)
    scale = chi2 / (n - 2)
    se = np.sqrt(scale * Sw / det)
    valid = ok & converged & np.isfinite(b) & np.isfinite(se) & (se > 0) & (np.abs(b) < 5)
    return b, se, valid


def check_against_statsmodels(Y, O, t, n_check=300):
    import statsmodels.api as sm
    b, se, valid = fit_quasipoisson(Y[:n_check], O[:n_check], t)
    worst = 0.0
    for i in range(n_check):
        if not valid[i]:
            continue
        f = sm.GLM(Y[i], sm.add_constant(t), family=sm.families.Poisson(),
                   offset=O[i]).fit(scale="X2")
        worst = max(worst, abs(float(f.params[1]) - b[i]), abs(float(f.bse[1]) - se[i]))
    return worst


# --------------------------------------------------------------------------- analytic
def analytic_power(D, b, c0, c1, centre, national_rate, years=N_YEARS):
    """Power of the unpooled test for a unit with D expected deaths per year.

    Poisson information with negative binomial variance (sandwich), alpha from the fitted
    exposure relation, non-central t on n-2 degrees of freedom.
    """
    t = np.arange(years) - (years - 1) / 2
    births = D / national_rate
    alpha = np.exp(c0 + c1 * (np.log(births) - centre))
    shape = np.exp(b * t)
    mu = D * shape / shape.mean()
    X = np.column_stack([np.ones(years), t])
    bread = np.linalg.inv(X.T @ (mu[:, None] * X))
    meat = X.T @ ((mu * (1 + mu / alpha))[:, None] * X)
    var_b = (bread @ meat @ bread)[1, 1]
    ncp = abs(b) / np.sqrt(var_b)
    df = years - 2
    upper = stats.nct.sf(T_CRIT, df, ncp)
    lower = stats.nct.cdf(-T_CRIT, df, ncp)
    # scipy returns NaN for the far tail at large non-centrality; that tail is zero.
    return float(np.nan_to_num(upper, nan=1.0) + np.nan_to_num(lower, nan=0.0))


def analytic_threshold(target, b, **kw):
    f = lambda logD: analytic_power(np.exp(logD), b, **kw) - target  # noqa: E731
    return float(np.exp(optimize.brentq(f, np.log(0.5), np.log(1e6))))


def interpolate_threshold(curve, target):
    """First crossing of `target` on a power curve, log-linear in exposure; None if never."""
    curve = curve.dropna(subset=["power"]).sort_values("deaths_per_unit_year")
    x, y = np.log(curve.deaths_per_unit_year.values), curve.power.values
    above = np.where(y >= target)[0]
    if len(above) == 0 or above[0] == 0:
        return None
    i = above[0]
    frac = (target - y[i - 1]) / (y[i] - y[i - 1])
    return float(np.exp(x[i - 1] + frac * (x[i] - x[i - 1])))


# --------------------------------------------------------------------------- main
def main():
    s = setup()
    panel, quint, pooled = s["panel"], s["quint"], s["pooled"]
    n_regions = len(s["regions"])
    idx = panel.region_idx.values
    t = np.sort(panel.t.unique())
    births = panel.pivot(index="region_idx", columns="t", values="births").values
    log_births = np.log(births)
    t_rows = panel.pivot(index="region_idx", columns="t", values="t").values
    assert np.allclose(t_rows, t[None, :])

    effects = {
        "national_8pct": s["mu_b"],
        "mde_14pct": -s["median_mde"],
        "decline_20pct": float(np.log(0.8) / 10),
        "null": 0.0,
    }
    decade = {k: round(100 * (np.exp(10 * v) - 1), 1) for k, v in effects.items()}
    deaths_per_year = pooled / N_YEARS
    kw = dict(c0=s["c0"], c1=s["c1"], centre=s["log_exp_centre"],
              national_rate=s["national_rate"])

    # Validate the vectorised fit once on real-scale null data.
    mu0 = births * np.exp(s["level"][:, None])
    a0 = s["alpha"][:, None]
    Y0 = RNG.negative_binomial(a0, a0 / (a0 + mu0))
    worst = check_against_statsmodels(Y0, log_births, t)
    assert worst < 1e-6, f"vectorised IRLS disagrees with statsmodels by {worst}"
    print(f"IRLS check against statsmodels: max abs difference {worst:.2e}")

    rows, series = [], []
    for k in SCALES:
        B = births * k
        logB = np.log(B)
        alpha_k = s["alpha"] * k ** s["c1"]            # dispersion along the fitted relation
        for name, b_true in effects.items():
            mu = B * np.exp(s["level"][:, None] + b_true * t[None, :])
            a = alpha_k[:, None]
            Y = RNG.negative_binomial(a, a / (a + mu), size=(SIMS, n_regions, N_YEARS))
            Y = Y.reshape(-1, N_YEARS)
            O = np.tile(logB, (SIMS, 1))
            bh, se, valid = fit_quasipoisson(Y, O, t)
            declared = valid & (np.abs(bh) > T_CRIT * se)
            region = np.tile(np.arange(n_regions), SIMS)
            df = pd.DataFrame({
                "region": region, "quintile": quint[region], "valid": valid,
                "declared": declared,
                "wrong_sign": declared & (b_true != 0) & (np.sign(bh) != np.sign(b_true)),
                "ratio": np.where(declared & (b_true != 0), np.abs(bh) / abs(b_true) if b_true else np.nan, np.nan),
                "deaths_per_unit_year": k * deaths_per_year[region],
            })
            df = df[df.valid]
            series.append(df.assign(effect=name, scale=k))
            for q, sub in df.groupby("quintile"):
                decl = sub[sub.declared]
                rows.append({
                    "effect": name, "decade_change_pct": decade[name], "true_slope": b_true,
                    "scale": k, "exposure_quintile": int(q),
                    "median_deaths_per_unit_year": round(float(k * np.median(deaths_per_year[quint == q])), 1),
                    "n": len(sub), "power": round(float(sub.declared.mean()), 3),
                    "type_S": round(float(decl.wrong_sign.mean()), 4) if len(decl) and b_true else np.nan,
                    "type_M": round(float(decl.ratio.median()), 2) if len(decl) and b_true else np.nan,
                    "analytic_power": round(analytic_power(
                        float(k * np.median(deaths_per_year[quint == q])), b_true, **kw), 3) if b_true else np.nan,
                })
        print(f"scale {k}x done")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "power_threshold_armA.csv", index=False)
    allseries = pd.concat(series, ignore_index=True)

    # Binned power curve on the common exposure axis, pooled over regions and scales.
    curves = []
    for name, b_true in effects.items():
        sub = allseries[allseries.effect == name].copy()
        sub["bin"] = pd.cut(sub.deaths_per_unit_year, BIN_EDGES)
        for b_, g in sub.groupby("bin", observed=True):
            x = float(np.exp(np.log(g.deaths_per_unit_year).mean()))
            curves.append({
                "effect": name, "bin_lo": b_.left, "bin_hi": b_.right,
                "deaths_per_unit_year": round(x, 2), "n": len(g),
                "power": round(float(g.declared.mean()), 4),
                "analytic_power": round(analytic_power(x, b_true, **kw), 4) if b_true else np.nan,
            })
    curve = pd.DataFrame(curves)
    curve.to_csv(OUT / "tables" / "power_threshold_curve.csv", index=False)

    ladder = pd.read_csv(OUT / "tables" / "aggregation_ladder.csv").set_index("level")
    thresholds = {}
    for name, b_true in effects.items():
        if not b_true:
            continue
        c = curve[curve.effect == name]
        thresholds[name] = {
            "true_slope_per_year": round(b_true, 5),
            "decade_change_pct": decade[name],
            **{f"simulated_deaths_per_unit_year_for_{int(100*p)}pct_power":
               (round(v, 1) if (v := interpolate_threshold(c, p)) is not None else None)
               for p in TARGETS},
            **{f"analytic_deaths_per_unit_year_for_{int(100*p)}pct_power":
               round(analytic_threshold(p, b_true, **kw), 1) for p in TARGETS},
        }
    # Which observed units already sit at or above each analytic 80% threshold.
    state_per_year = panel.groupby("UF")[OUTCOME].sum() / N_YEARS
    for name, thr in thresholds.items():
        d = thr["analytic_deaths_per_unit_year_for_80pct_power"]
        thr["observed_units_at_or_above_80pct_threshold"] = {
            "immediate_regions_of_510": int((deaths_per_year >= d).sum()),
            "states_of_27": int((state_per_year >= d).sum()),
        }
    # Largest type S among declarations at or above 240 deaths per unit-year (national slope).
    nat = table[(table.effect == "national_8pct") & (table.median_deaths_per_unit_year >= 240)]
    null = allseries[allseries.effect == "null"].groupby("scale").declared.mean()
    summary = {
        "outcome": OUTCOME,
        "sims_per_region_per_scenario": SIMS,
        "scales": list(SCALES),
        "max_simulated_deaths_per_unit_year": round(float(allseries.deaths_per_unit_year.max()), 1),
        "series_dropped_invalid_pct": round(100 * (1 - len(allseries) / (SIMS * n_regions * len(SCALES) * len(effects))), 3),
        "null_declaration_rate_by_scale": {int(k): round(float(v), 4) for k, v in null.items()},
        "null_declaration_rate_overall": round(float(allseries[allseries.effect == "null"].declared.mean()), 4),
        "thresholds": thresholds,
        "national_slope_max_type_S_at_or_above_240_per_unit_year": round(float(nat.type_S.max()), 4),
        "inputs": {
            "mu_b": round(s["mu_b"], 5), "median_mde_pooled": round(s["median_mde"], 5),
            "log_alpha0": round(s["c0"], 4), "log_alpha_slope": round(s["c1"], 4),
            "log_exposure_centre": round(s["log_exp_centre"], 4),
            "national_rate_deaths_per_birth": round(s["national_rate"], 6),
            "t_critical_9df": round(float(T_CRIT), 4),
        },
        "observed_deaths_per_unit_year": {
            "quintile_medians_immediate_region": [round(float(np.median(deaths_per_year[quint == q])), 1) for q in range(1, 6)],
            **{lvl: round(float(ladder.loc[lvl, "median_deaths_per_unit"]) / N_YEARS, 1) for lvl in ladder.index},
            "largest_immediate_region": round(float(deaths_per_year.max()), 1),
            "largest_state": round(float(panel.groupby("UF")[OUTCOME].sum().max()) / N_YEARS, 1),
        },
    }
    (OUT / "POWER_THRESHOLD.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    make_figure(curve, thresholds, summary["observed_deaths_per_unit_year"], effects, decade, kw)


# --------------------------------------------------------------------------- figure
def make_figure(curve, thresholds, observed, effects, decade, kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"national_8pct": "#2a78d6", "mde_14pct": "#eb6834", "decline_20pct": "#1baf7a"}
    labels = {"national_8pct": f"{abs(decade['national_8pct']):.1f}% per decade (national slope)",
              "mde_14pct": f"{abs(decade['mde_14pct']):.1f}% per decade (median detectable slope, pooled)",
              "decline_20pct": f"{abs(decade['decline_20pct']):.0f}% per decade (20% scenario)"}
    grid = np.exp(np.linspace(np.log(2), np.log(20000), 200))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.set_facecolor("white")
    q = observed["quintile_medians_immediate_region"]
    ax.axvspan(q[0], q[-1], color="#e7e6e2", lw=0, zorder=0)
    ax.text(np.sqrt(q[0] * q[-1]), 0.97, "observed immediate regions\n(quintile medians)",
            ha="center", va="top", fontsize=7, color="#52514e")
    for lvl, lab in (("state", "median state"), ("macro-region", "median macro-region")):
        ax.axvline(observed[lvl], color="#b5b4ae", lw=1, ls=":", zorder=1)
        ax.text(observed[lvl] * 1.05, 0.03, lab, rotation=90, fontsize=7, color="#52514e", va="bottom")
    ax.axhline(0.8, color="#b5b4ae", lw=1, ls="--", zorder=1)
    ax.text(2.1, 0.815, "80% power", fontsize=7, color="#52514e")
    ax.axhline(0.05, color="#b5b4ae", lw=1, ls="--", zorder=1)
    ax.text(19000, 0.065, "5% false-positive rate", fontsize=7, color="#52514e", ha="right")

    for name, b_true in effects.items():
        if not b_true:
            continue
        y = [analytic_power(x, b_true, **kw) for x in grid]
        ax.plot(grid, y, color=colours[name], lw=2, label=labels[name], zorder=3)
        c = curve[curve.effect == name]
        ax.scatter(c.deaths_per_unit_year, c.power, s=16, color="white", edgecolor=colours[name],
                   lw=1.2, zorder=4)
        thr = thresholds[name]["analytic_deaths_per_unit_year_for_80pct_power"]
        ax.annotate(f"{thr:,.0f}", xy=(thr, 0.8), xytext=(thr, 0.86), ha="center", fontsize=7,
                    color=colours[name], zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(2, 20000)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{LABEL.capitalize()} per unit-year (expected)")
    ax.set_ylabel("Power, unpooled fit of an 11-year series")
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color="#eeede9", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.legend(loc="center left", fontsize=7, frameon=False, title="True change", title_fontsize=7)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "power_threshold.png")
    fig.savefig(OUT / "figures" / "power_threshold.pdf")
    print("wrote figures/power_threshold.png and .pdf")


if __name__ == "__main__":
    main()
