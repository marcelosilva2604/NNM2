"""Audit of three things the earlier review rounds questioned, using stored posteriors only.

No model is refitted. Everything is read from 2-model/allcause (all-cause neonatal deaths).

1. Aggregate national trend. The manuscript calls the hierarchical slope hypermean mu_b the
   "national slope". The trend of the aggregate national rate, sum of expected deaths over
   sum of livebirths per year, is a different estimand: it weights regions by births and by
   their own level. Here it is computed draw by draw from the region intercepts and slopes
   and compared with mu_b and with the observed aggregate series.
2. Stability of the classified regions across the four slope-prior specifications of the
   spatial-robustness analysis (no pooling, exchangeable, scaled BYM2 with the state term) and the primary model:
   which regions are classified, how much the sets overlap, how the direction probability
   P(slope < 0) compares, and where the primary classified regions are.
3. Completeness algebra. With observed rate = true rate * capture_deaths / capture_births,
   the observed log-slope equals the true log-slope plus g_D minus g_B (annual log changes in
   death and birth capture). Table of the true slope implied by an observed slope of the
   size reported, for a grid of capture trends.

Run:
    NNM2_OUTCOME=deaths_total .venv/bin/python 3-results/26_audit_aggregate_trend_regions.py
Outputs go to 3-results/allcause/audit_2026-10-17/.
"""
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, model_dir, results_dir  # noqa: E402

assert OUTCOME == "deaths_total", "run with NNM2_OUTCOME=deaths_total"
MODEL = model_dir(ROOT)
OUT = results_dir(ROOT) / "audit_2026-10-17"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = ROOT / "data" / "processed" / "panel_region_year.csv"
THIN = int(__import__("os").environ.get("NNM2_THIN", 1))  # 1 = all posterior draws


def draws(idata, name):
    """Posterior draws as array (units, samples)."""
    return idata.posterior[name].stack(sample=("chain", "draw")).values


def ols_slope(y, x):
    """OLS slope of y (len(x), n) on x, per column."""
    xc = x - x.mean()
    return (xc[:, None] * (y - y.mean(0))).sum(0) / (xc ** 2).sum()


def interval(v):
    return [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


log = {}

# ---------------------------------------------------------------- panel, order of regions
df = pd.read_csv(PANEL).sort_values(["rgi_id", "year"]).reset_index(drop=True)
regions = np.sort(df.rgi_id.unique())
years = np.sort(df.year.unique())
t = (years - years.mean()).astype(float)
B = df.pivot(index="rgi_id", columns="year", values="births").loc[regions, years].values.astype(float)
D = df.pivot(index="rgi_id", columns="year", values="deaths_total").loc[regions, years].values.astype(float)
assert B.shape == (510, 11), B.shape

prim_slopes = pd.read_csv(MODEL / "slopes_rate.csv")
assert (prim_slopes.rgi_id.values == regions).all(), "region order differs from the slopes table"

# ---------------------------------------------------------------- 1. aggregate national trend
idata = az.from_netcdf(MODEL / "idata_rate.nc")
a = draws(idata, "a")[:, ::THIN]
b = draws(idata, "b")[:, ::THIN]
mu_b = idata.posterior["mu_b"].stack(sample=("chain", "draw")).values[::THIN]
n = a.shape[1]

# expected deaths per region-year and draw: births * exp(a + b t); the model has no extra offset
E = B[:, :, None] * np.exp(a[:, None, :] + b[:, None, :] * t[None, :, None])
agg_rate = E.sum(0) / B.sum(0)[:, None]          # (years, draws)
agg_slope = ols_slope(np.log(agg_rate), t)
w_births = B.sum(1) / B.sum()
w_deaths = D.sum(1) / D.sum()
wb_births = (w_births[:, None] * b).sum(0)       # births-weighted mean of region slopes
wb_deaths = (w_deaths[:, None] * b).sum(0)

obs_rate = D.sum(0) / B.sum(0)
obs_slope = float(ols_slope(np.log(obs_rate)[:, None], t)[0])
try:
    import statsmodels.api as sm
    X = sm.add_constant(t)
    pois = sm.GLM(D.sum(0), X, family=sm.families.Poisson(), offset=np.log(B.sum(0))).fit(
        cov_type="HC0")
    obs_poisson = [float(pois.params[1]), float(pois.conf_int()[1, 0]), float(pois.conf_int()[1, 1])]
except Exception as exc:  # statsmodels is optional
    obs_poisson = f"not computed ({exc})"

log["aggregate_trend"] = {
    "draws_used": int(n),
    "mu_b_hypermean": interval(mu_b),
    "aggregate_rate_log_slope": interval(agg_slope),
    "births_weighted_mean_region_slope": interval(wb_births),
    "deaths_weighted_mean_region_slope": interval(wb_deaths),
    "unweighted_mean_region_slope": interval(b.mean(0)),
    "observed_aggregate_log_rate_ols_slope": obs_slope,
    "observed_aggregate_poisson_slope_with_robust_95ci": obs_poisson,
    "pct_change_per_decade": {
        "mu_b": float(100 * (np.exp(10 * np.mean(mu_b)) - 1)),
        "aggregate": float(100 * (np.exp(10 * np.mean(agg_slope)) - 1)),
        "observed_ols": float(100 * (np.exp(10 * obs_slope) - 1)),
    },
    "difference_aggregate_minus_mu_b": interval(agg_slope - mu_b),
}
del E

# ---------------------------------------------------------------- 2. regional stability
specs = {"primary_three_level": ("rate", "idata_rate.nc"),
         "nopool": ("spatial_nopool", "idata_spatial_nopool.nc"),
         "exchangeable": ("spatial_exch", "idata_spatial_exch.nc"),
         "bym2_scaled": ("spatial_bym2_scaled_long", "idata_spatial_bym2_scaled_long.nc")}
tab = pd.DataFrame({"rgi_id": regions, "UF": df.drop_duplicates("rgi_id").set_index("rgi_id").loc[regions, "UF"].values})
for name, (stem, fn) in specs.items():
    idt = idata if name == "primary_three_level" else az.from_netcdf(MODEL / fn)
    bb = draws(idt, "b")
    tab[f"{name}_mean"] = bb.mean(1)
    tab[f"{name}_sd"] = bb.std(1)
    tab[f"{name}_pneg"] = (bb < 0).mean(1)
    lo, hi = np.percentile(bb, 2.5, axis=1), np.percentile(bb, 97.5, axis=1)
    tab[f"{name}_classified"] = (lo > 0) | (hi < 0)
    tab[f"{name}_decline"] = hi < 0
    tab[f"{name}_increase"] = lo > 0
    tab[f"{name}_width"] = hi - lo
    if name != "primary_three_level":
        del idt
tab.to_csv(OUT / "region_by_specification.csv", index=False)

names = list(specs)
overlap = {}
for i, x in enumerate(names):
    for y in names[i + 1:]:
        sx, sy = tab[f"{x}_classified"].values, tab[f"{y}_classified"].values
        inter, uni = int((sx & sy).sum()), int((sx | sy).sum())
        overlap[f"{x} vs {y}"] = {"n_x": int(sx.sum()), "n_y": int(sy.sum()), "both": inter,
                                   "jaccard": round(inter / uni, 3) if uni else None,
                                   "spearman_slope_means": round(float(stats.spearmanr(tab[f"{x}_mean"], tab[f"{y}_mean"])[0]), 3)}
all4 = tab[[f"{k}_classified" for k in names]].all(axis=1)
any4 = tab[[f"{k}_classified" for k in names]].any(axis=1)
prim = tab.primary_three_level_classified
by_uf = (tab.assign(cls=prim).groupby("UF").agg(regions=("rgi_id", "size"), classified=("cls", "sum"))
         .assign(share=lambda d: (d.classified / d.regions).round(3)).sort_values("classified", ascending=False))
by_uf.to_csv(OUT / "primary_classified_by_state.csv")
log["regional_stability"] = {
    "classified_by_spec": {k: int(tab[f"{k}_classified"].sum()) for k in names},
    "declines_by_spec": {k: int(tab[f"{k}_decline"].sum()) for k in names},
    "increases_by_spec": {k: int(tab[f"{k}_increase"].sum()) for k in names},
    "classified_in_all_four": int(all4.sum()),
    "classified_in_at_least_one": int(any4.sum()),
    "pairwise_overlap": overlap,
    "primary_classified_by_state_top": by_uf.head(10).reset_index().to_dict("records"),
    "states_with_any_primary_classified": int((by_uf.classified > 0).sum()),
    "primary_pneg_ge_0.95": int((tab.primary_three_level_pneg >= 0.95).sum()),
    "primary_pneg_ge_0.90": int((tab.primary_three_level_pneg >= 0.90).sum()),
    "primary_pneg_le_0.5_regions_with_positive_mean": int((tab.primary_three_level_mean > 0).sum()),
    "median_interval_width": {k: round(float(tab[f"{k}_width"].median()), 4) for k in names},
}

# ---------------------------------------------------------------- 3. completeness algebra
s_obs_mu = float(np.mean(mu_b))
s_obs_agg = float(np.mean(agg_slope))
rows = []
for gD in (-0.0100, -0.0084, -0.0050, 0.0, 0.0050, 0.0084, 0.0100):
    for label, s_obs in (("mu_b", s_obs_mu), ("aggregate", s_obs_agg)):
        true = s_obs - gD  # g_B = 0
        rows.append({"observed_slope_kind": label, "observed_slope": round(s_obs, 5),
                     "death_capture_annual_log_change": gD, "implied_true_slope": round(true, 5),
                     "implied_true_pct_change_per_decade": round(100 * (np.exp(10 * true) - 1), 1)})
pd.DataFrame(rows).to_csv(OUT / "completeness_algebra.csv", index=False)
log["completeness"] = {
    "relation": "observed_slope = true_slope + g_D - g_B",
    "death_capture_change_that_would_create_the_observed_slope_from_a_flat_truth": round(s_obs_agg, 5),
    "note": "improving death capture (g_D > 0) implies a STEEPER true decline; a deterioration of about "
            f"{abs(s_obs_agg):.4f} per year in relative death capture could produce the observed decline from a flat truth.",
}

json.dump(log, open(OUT / "audit_summary.json", "w"), indent=2, default=float)
print(json.dumps(log, indent=2, default=float))
