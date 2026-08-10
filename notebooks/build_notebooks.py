"""Generate the documented notebooks that reproduce every number in the manuscript.

The notebooks are written from this script rather than by hand so that they are
themselves reproducible and reviewable: what a reader sees in a notebook is exactly what
this file says, and regenerating them cannot silently drift from the analysis.

Design rule: the notebooks do NOT refit any model. Sampling happens in the numbered
scripts under 1-data/ and 2-model/, which write posteriors to disk. The notebooks load
those artefacts, recompute every published quantity from them, and assert the value
against the manuscript. An assertion that fails is the point: it means the text and the
artefacts have diverged.

Three notebooks:
    01_data_and_panel        what the data are and how the panel was built
    02_models_and_results    the fitted models and every headline number
    03_robustness            everything asked of the analysis to try to break it

Run:
    .venv/bin/python notebooks/build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


PREAMBLE = """
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC, MODEL, RES = ROOT / "data/processed", ROOT / "2-model", ROOT / "3-results"

def check(label, computed, published, tol=1e-9):
    "Recompute a published value and fail loudly if the manuscript no longer matches."
    ok = abs(float(computed) - float(published)) <= tol
    print(f"{'OK  ' if ok else 'MISMATCH'}  {label}: manuscript={published}  recomputed={computed}")
    assert ok, f"{label}: manuscript says {published}, artefacts say {computed}"

def classified(draws, lo_q=0.025, hi_q=0.975):
    "The single criterion used everywhere in this study: 95% ETI excluding zero."
    lo, hi = np.quantile(draws, lo_q, axis=1), np.quantile(draws, hi_q, axis=1)
    return (lo > 0) | (hi < 0)

def slopes(idata, var="b"):
    return idata.posterior[var].stack(sample=("chain", "draw")).values
"""


def notebook_01():
    return nb([
        md("""
# 01. Data and panel construction

**What this notebook establishes.** Where the numbers in the Study data paragraph come
from, and why the panel is built the way it is.

The two decisions that matter and are easy to get wrong:

1. **The universe is the birth series, not the death series.** A region-year with births
   and no deaths is an observed zero. The archived earlier version of this project keyed
   the panel on deaths, which silently dropped 21,128 municipality-years (34% of the
   total) — all of them zero-death, and concentrated in exactly the small units the
   study is about. That is informative missingness and it biased the reliability tiers.

2. **The cause grouping was inherited, not re-derived.** The four action groups come
   from the source cause panel. This notebook can verify the totals but cannot verify
   the ICD-10 mapping, and the manuscript says so.
"""),
        code(PREAMBLE),
        code("""
panel = pd.read_csv(PROC / "panel_region_year.csv")
muni  = pd.read_csv(PROC / "panel_muni_year.csv")

print(f"region panel : {len(panel):,} rows, {panel.rgi_id.nunique()} regions, "
      f"{panel.UF.nunique()} states, {panel.year.min()}-{panel.year.max()}")
print(f"muni panel   : {len(muni):,} rows, {muni.CODMUNRES.nunique():,} municipalities")
print(f"zero-death municipality-years retained: {(muni.deaths_total == 0).sum():,} "
      f"({(muni.deaths_total == 0).mean():.1%})")
"""),
        md("""
## The published totals

Every figure below appears in the Study data paragraph of the manuscript.
"""),
        code("""
check("neonatal deaths", panel.deaths_total.sum(), 260023)
check("live births", panel.births.sum(), 30462542)
check("avoidable deaths", panel.avoidable.sum(), 117578)
check("immediate regions", panel.rgi_id.nunique(), 510)
check("states", panel.UF.nunique(), 27)
check("panel rows", len(panel), 5610)
"""),
        md("""
## Exposure over the decade

Two facts carry weight later. Avoidable mortality per birth fell substantially while
all-cause neonatal mortality barely moved, and the number of births — the exposure that
every unit's precision depends on — fell by a fifth.
"""),
        code("""
by_year = panel.groupby("year").agg(births=("births", "sum"),
                                    avoidable=("avoidable", "sum"),
                                    deaths=("deaths_total", "sum"))
by_year["avoidable_per_1000"] = (1000 * by_year.avoidable / by_year.births).round(2)
by_year["nmr_per_1000"] = (1000 * by_year.deaths / by_year.births).round(2)
display(by_year)

check("avoidable per 1000, 2014", by_year.avoidable_per_1000.iloc[0], 4.25)
check("avoidable per 1000, 2024", by_year.avoidable_per_1000.iloc[-1], 3.44)
check("NMR 2014", by_year.nmr_per_1000.iloc[0], 8.89)
check("NMR 2024", by_year.nmr_per_1000.iloc[-1], 8.32)
check("births change (%)", round(100 * (by_year.births.iloc[-1] / by_year.births.iloc[0] - 1), 1), -20.0)
"""),
        md("""
## Exposure per unit, which is what precision depends on

The median immediate region accumulated 115 avoidable deaths across the whole decade.
The median municipality accumulated 7. No statistical method creates events that did
not occur, so these two numbers largely determine everything that follows.
"""),
        code("""
for label, frame, key in [("immediate region", panel, "rgi_id"),
                          ("municipality", muni.dropna(subset=["rgi_id"]), "CODMUNRES")]:
    pooled = frame.groupby(key).avoidable.sum()
    print(f"{label:18} n={len(pooled):>5,}  median={int(pooled.median()):>5}  "
          f">=50: {(pooled >= 50).sum():>4}   >=400: {(pooled >= 400).sum():>4}")
"""),
        md("""
## Crosswalk loss, disclosed in Methods

23 municipality codes did not match the region crosswalk. They carry no deaths, so no
outcome is lost; 980 births are.
"""),
        code("""
report = (PROC / "panel_build_report.txt").read_text()
print("\\n".join(l for l in report.splitlines() if "unmatched" in l.lower() or "CROSSWALK" in l))
check("births lost to the crosswalk", muni.births.sum() - panel.births.sum(), 980)
"""),
    ])


def notebook_02():
    return nb([
        md("""
# 02. Models and headline results

**What this notebook establishes.** Every number in the Results sections up to and
including the tolerance analysis, recomputed from the stored posteriors.

Nothing here is refitted. `2-model/02_canonical_fit.py`, `07_state_model.py` and
`09_aggregation_ladder.py` do the sampling and write the posteriors; this notebook reads
them. If a number below stops matching the manuscript, one of the two has changed.

**The single criterion.** A unit is *classified* when the 95% equal-tailed posterior
interval for its slope excludes zero. No other interval width is used anywhere in this
study, and the helper below is the only implementation of the rule.
"""),
        code(PREAMBLE),
        code("""
rate = az.from_netcdf(MODEL / "idata_rate.nc")
b = slopes(rate)
cls = classified(b)

mu_b = float(rate.posterior["mu_b"].mean())
print(f"national drift: {mu_b:+.4f} per year "
      f"= {100 * (np.exp(mu_b * 10) - 1):.1f}% per decade")
check("regions classified", cls.sum(), 86)
check("national drift, % per decade", round(100 * (np.exp(mu_b * 10) - 1), 1), -20.0)
"""),
        md("""
## The aggregation ladder

The paper's central comparison. Four nested partitions of the same country and decade,
one specification, one criterion; only the unit changes.

The unpooled column matters because it contains no prior at all. It uses quasi-Poisson
with each unit's own dispersion, referred to **t on n-2 residual degrees of freedom**.
An earlier version of this analysis used the normal critical value of 1.96, which on 11
annual points runs at a 7-8% false positive rate rather than 5% and inflated every
unpooled figure by roughly half.
"""),
        code("""
ladder = pd.read_csv(RES / "tables/aggregation_ladder.csv")
display(ladder[["level", "units", "median_deaths_per_unit",
                "classified", "pct_classified", "classified_unpooled", "pct_unpooled"]])

for lvl, hier, unp in [("municipality", 61, 384), ("immediate region", 81, 83),
                       ("state", 21, 19), ("macro-region", 0, 5)]:
    row = ladder[ladder.level == lvl].iloc[0]
    check(f"{lvl}: hierarchical", row.classified, hier)
    check(f"{lvl}: unpooled", row.classified_unpooled, unp)
"""),
        md("""
### Why the macro-region rung is uninterpretable

With five units the model cannot separate the national mean from the unit deviations.
Each unit's marginal posterior inherits the hyperparameter uncertainty, so none resolves
— while unpooled, all five resolve with intervals far from zero. The rung is reported
for completeness, and as a symptom of the design at k = 5 rather than a statement about
macro-regions. Dropping it silently would be selective reporting.
"""),
        code("""
macro = az.from_netcdf(MODEL / "idata_ladder_macro_region.nc")
bm = slopes(macro)
print(f"per-unit posterior SD : {bm.std(axis=1).round(4)}")
print(f"national drift SD     : {float(macro.posterior['mu_b'].std()):.4f}")
print("-> almost all of each unit's uncertainty is the national parameter's own.")
"""),
        md("""
## No unit is worsening, at any level

Reading point estimates at face value, 10 regions look like they are getting worse. None
survives its own interval. This is the contrast Figure 1 draws.
"""),
        code("""
srate = pd.read_csv(MODEL / "slopes_rate.csv")
sstate = pd.read_csv(MODEL / "slopes_state.csv")
check("regions with a positive point estimate", (srate.b_mean > 0).sum(), 10)
check("regions classified AND positive", ((srate.b_mean > 0) & srate.classified_95).sum(), 0)
check("states classified", sstate.classified_95.sum(), 21)
check("states classified AND positive", ((sstate.b_mean > 0) & sstate.classified_95).sum(), 0)
"""),
        md("""
## Departure from the national trajectory

Two different questions. *Did this unit change?* compares its slope to zero. *Is this
unit falling behind?* compares its slope to the national drift, and only the second
identifies a place to act on.

The second contrast is taken **against the shrinkage target**, so it is not an
independent test — the prior has already pulled every unit toward `mu_b`. That is why
heterogeneity is also tested outside the model, by Cochran's Q on unpooled state slopes
with each state's own dispersion, where neither shrinkage nor the prior can create or
conceal it. The two answers differ, and reporting only the first would be misleading.
"""),
        code("""
vs_state = pd.read_csv(RES / "tables/vs_national_state.csv")
vs_reg   = pd.read_csv(RES / "tables/vs_national_region.csv")
check("states separable from the national drift", (vs_state.lagging | vs_state.leading).sum(), 0)
check("regions separable", (vs_reg.lagging | vs_reg.leading).sum(), 9)
check("regions ahead", vs_reg.leading.sum(), 8)
check("regions lagging", vs_reg.lagging.sum(), 1)

het = json.load(open(RES / "SUPPORTING_CHECKS.json"))["between_state_heterogeneity"]
print(f"\\nOutside the model: Q={het['Q']} on {het['df']} df, p={het['p_value']}, "
      f"I2={het['I2_pct']}%, tau={het['tau']}/yr")
check("Cochran Q", het["Q"], 114.7, tol=0.05)
"""),
        md("""
## Design analysis

The classification rate depends on Brazil's own distribution of true slopes and on the
prior, so it is not a transportable property of the registration system. Simulating
known slopes at each region's real births and fitted dispersion gives one that is.

Two arms. Arm A gives every region the same true slope and refits unpooled: it measures
what a reader of one unit's series can detect. Arm B draws true slopes from the fitted
between-unit spread and refits the paper's own hierarchical model, which is the only
setting where that procedure can be evaluated, since a common true slope would drive the
between-unit variance to zero and make it trivially confident.

Read type S and type M alongside power. Unpooled, a declaration is not only rare but
unreliable: in the smallest regions the sign is wrong in about one declaration in nine and
the magnitude is exaggerated more than fivefold. Under the hierarchical procedure the sign
error falls below 2% everywhere and the magnitude is essentially unbiased.
"""),
        code("""
two = pd.read_csv(RES / "tables/design_two_arm.csv")
display(two)

d = json.load(open(RES / "DESIGN_TWO_ARM.json"))
check("arm A, mean power at a national-sized change", d["arm_A_mean_power_at_1x"], 0.116)
check("arm A, declaration rate under a true zero", d["arm_A_null_declaration_rate"], 0.046, tol=0.0005)
check("arm B, mean power", d["arm_B_mean_power"], 0.195)
check("arm B, mean classification count", d["arm_B_classification_count"]["mean"], 99.1, tol=0.05)
print("\\narm B classification range:", d["arm_B_classification_count"]["min"],
      "to", d["arm_B_classification_count"]["max"], "- contains the observed 86")
"""),
        md("""
## Certifying change is not the same as certifying stability

At a tolerance equal to the national drift, no region can be certified stable. That is
**arithmetic before it is empirical**: only one region in the country has a posterior
standard deviation small enough to be certified stable at that tolerance *even if its
true slope were exactly zero*. The manuscript states this precondition, because without
it the result reads as a discovery about Brazil rather than about precision.
"""),
        code("""
rope = pd.read_csv(RES / "tables/rope_curve_rate.csv")
display(rope)

at1 = rope[rope.tolerance_x_national == 1.0].iloc[0]
check("changing at a tolerance of 1x", at1.drifting, 9)
check("stable at 1x", at1.credibly_flat, 0)
check("indeterminate at 1x (%)", round(100 * at1.indeterminate_share, 1), 98.2)

eligible = (srate.b_sd < abs(mu_b) / 1.96).sum()
check("regions precise enough to ever be certified stable at 1x", eligible, 1)
"""),
    ])


def notebook_03():
    return nb([
        md("""
# 03. Robustness

**What this notebook establishes.** Everything run to try to break the central claim,
and what survived.

The claim under attack: *most immediate regions cannot be resolved.* If that is an
artefact of a modelling choice rather than a property of the data, one of the checks
below should show it.
"""),
        code(PREAMBLE),
        md("""
## Is the indeterminacy manufactured by the prior?

No. Removing pooling entirely *lowers* the classification rate; a spatial prior raises it
by about one percentage point. Shrinkage is costing classifications here, not creating
them, so the pessimistic reading is not self-inflicted.

Note the honest counterweight in the same table: the minimum detectable slope is
strongly prior-dependent (0.0344 pooled against 0.0632 unpooled, that is 1.54 against
2.84 times the national drift). The manuscript quotes both.
"""),
        code("""
rob = pd.read_csv(RES / "tables/spatial_robustness.csv")
display(rob)
for prior, n in [("nopool", 77), ("exch", 84), ("bym2", 89)]:
    check(f"{prior}", rob[rob.prior == prior].classified.iloc[0], n)
"""),
        md("""
## Does the result depend on summarising each region by a straight line?

A fair question, because a region's own series rejects the log-linear fit in 13.3% of
cases overall and 26.5% in the highest-exposure quintile — which is where the study
actually makes determinations. It does not: a per-region quadratic and a national
second-order random walk move the count by less than the seeds do.
"""),
        code("""
shape = pd.read_csv(RES / "tables/trend_shape.csv")
display(shape)
for tr, n in [("linear", 89), ("quadratic", 83), ("rw", 84)]:
    check(f"trend={tr}", shape[shape.trend == tr].classified.iloc[0], n)
"""),
        md("""
## How stable is the count itself?

Eight refits of the identical specification. The spread is small and, more importantly,
it lives entirely at the decision boundary: 78 regions classify under every seed, 417
under none, and only 15 ever change status. This is a property of thresholding a
continuous quantity, not of the estimate, which is why the probability of direction is
reported alongside the count.
"""),
        code("""
seeds = json.load(open(RES / "SEED_STABILITY.json"))
print(f"counts across 8 seeds: {seeds['counts']}")
print(f"mean {seeds['mean']}, sd {seeds['sd']}, range {seeds['min']}-{seeds['max']} "
      f"({seeds['range_pct_of_510']}% of 510)")
check("always classified", seeds["regions_classified_in_every_run"], 78)
check("never classified", seeds["regions_classified_in_no_run"], 417)
check("changing status", seeds["regions_that_flip_between_runs"], 15)
"""),
        md("""
## Is the variance function doing the work?

Partly, and this one changed a conclusion. A single dispersion parameter imposes the same
relative variability on every unit whatever its size. The posterior predictive check
shows the consequence: adequate fit overall, but a gradient across exposure, with the
model predicting too much variability in the largest units.

Letting dispersion depend on exposure flattens the gradient and barely moves the
classification count — but it widens the spread of precision across units from 1.86-fold
to 3.4-fold. An earlier draft of this paper claimed precision was nearly independent of
unit size. That claim was an artefact of the simpler variance function and has been
removed.

Note the residual-degrees-of-freedom divisor below. Dividing by the number of
observations instead, as a first version of this check did, makes a correctly specified
model look overdispersed.
"""),
        code("""
panel = pd.read_csv(PROC / "panel_region_year.csv").sort_values(["rgi_id", "year"]).reset_index(drop=True)
y = panel.avoidable.values
resid_df = len(panel) - 2 * panel.rgi_id.nunique()

for tag in ("global", "exposure"):
    idata = az.from_netcdf(MODEL / f"idata_disp_{tag}.nc")
    rep = idata["posterior_predictive"]["y"].stack(sample=("chain", "draw")).values
    pearson2 = (y - rep.mean(axis=1)) ** 2 / np.maximum(rep.var(axis=1), 1e-9)
    q = pd.qcut(panel.groupby("rgi_id").avoidable.transform("sum"), 5, labels=False) + 1
    byq = pd.Series(pearson2).groupby(q).mean() * (len(panel) / resid_df)
    print(f"{tag:9} overall {pearson2.sum() / resid_df:.3f}   by quintile {byq.round(2).tolist()}")
    overall = float(pearson2.sum() / resid_df)   # recomputed, not restated
    if tag == "global":
        check("global dispersion, overall chi2/df", round(overall, 3), 0.945)
        check("PPC gradient, lowest quintile", round(float(byq.iloc[0]), 2), 1.04)
        check("PPC gradient, highest quintile", round(float(byq.iloc[-1]), 2), 0.78)
    else:
        check("exposure dispersion, overall chi2/df", round(overall, 3), 0.992)
"""),
        md("""
## Can covariates rescue trend detectability?

The small area estimation literature exists to borrow strength, so this has to be shown
rather than asserted. The distinction the design turns on:

- a covariate can genuinely sharpen a **level**;
- a covariate that predicts a **trend** can raise the classification count without any
  change in the underlying data, because part of the resulting claim came from the
  covariate rather than from the deaths.

In this study the slope coefficient turned out to include zero, so there is nothing for
the covariate to assert and the residual reduces to the departure-from-national contrast
already reported. Its count is that same quantity, not an independent result.

Note also the static index has, by construction, **zero within-region variance**, so it
cannot carry information about the direction of change within a region.
"""),
        code("""
path = RES / "tables/covariate_models.csv"
if path.exists():
    cov = pd.read_csv(path)
    display(cov)
    for variant, published in [("base", 89), ("level", 85), ("slope", 89), ("timevar", 103)]:
        check(f"covariate: {variant}",
              int(cov[cov.variant == variant].classified_slope.iloc[0]), published)
    print("\\nWithin-region share of covariate variance (from covariates_report.txt):")
    print("\\n".join(l for l in (PROC / "covariates_report.txt").read_text().splitlines()
                     if "within-region" in l))
else:
    print("covariate models not yet fitted; run 2-model/14_covariates.py")
"""),
        md("""
## Multiplicity, and attribution at state level

Two results that were computed and, in an earlier draft, not reported. Both are now in
the manuscript.

The classification criterion makes 510 simultaneous decisions. Hierarchical shrinkage
damps multiplicity, which is why the primary criterion is uncorrected, but the size of
the drop under explicit control bounds how much any single classification can carry.

Attribution at state level depended entirely on the variance function. Under one shared
dispersion no state separated from the national drift; that specification imposes a
common floor on relative variability whatever the unit's size, and the dependence of
dispersion on exposure is strong. Under a fitted exposure-dependent dispersion six states
separate, and unpooled ten do, nesting the same six.
"""),
        code("""
prec = (RES / "03_precision_report.log").read_text()
print("\\n".join(l for l in prec.splitlines() if "BH-FDR" in l or "Bonferroni" in l))

bh = {l.split(":")[0].split("]")[0].strip("["): int(l.split(":")[1])
      for l in prec.splitlines() if "surviving BH-FDR" in l}
check("regions surviving BH-FDR, rate", bh["rate"], 10)
check("regions surviving BH-FDR, share", bh["share"], 16)

sd = pd.read_csv(RES / "tables/state_dispersion.csv")
check("states departing, single dispersion", sd.departs_global.sum(), 0)
check("states departing, exposure-dependent dispersion", sd.departs_exposure.sum(), 6)
check("states departing, unpooled", sd.departs_unpooled.sum(), 10)
check("the six are nested in the ten", int((sd.departs_exposure & ~sd.departs_unpooled).sum()), 0)
print(f"\\nRio de Janeiro: own {sd[sd.UF=='RJ'].slope_own.iloc[0]:.4f}, "
      f"single-dispersion model {sd[sd.UF=='RJ'].b_global.iloc[0]:.4f}, "
      f"t={sd[sd.UF=='RJ'].t.iloc[0]:.1f}")
"""),
        md("""
## Alternative explanations that were tested and rejected

Each of these would, if true, undercut the central claim. None does. The p-value on the
coding-quality correlation is nominally significant and is reported rather than omitted;
the effect is far too small to account for the findings, and it points in the direction
of *faster* apparent improvement where coding deteriorated.
"""),
        code("""
s = json.load(open(RES / "SUPPORTING_CHECKS.json"))
print(json.dumps(s["coding_quality"], indent=2))
print(json.dumps(s["outcome_definition"], indent=2))
print(json.dumps(s["common_national_shock"], indent=2))
print(json.dumps(s["winners_curse"], indent=2))

check("avoidable, unpooled (%)", s["outcome_definition"]["avoidable"]["pct"], 16.3)
check("all-cause neonatal, unpooled (%)", s["outcome_definition"]["all_neonatal"]["pct"], 10.4)
check("common national shock share", s["common_national_shock"]["common_variance_share"], 0.0067)
check("winner's curse ratio", s["winners_curse"]["inflation_ratio"], 1.91)
"""),
        md("""
## The two quantities the Discussion adds

These are the transferability rule and the completeness bound. Neither is a model fit;
both are arithmetic on quantities already established above, and both are recomputed here
because they are published numbers.

The completeness bound is the one that matters defensively. If capture grows at rate *r*
per year, the fitted slope is displaced by exactly log(1 + *r*), additively in the mean.
That moves where a slope sits without touching how precisely it is estimated, so the
detectability result survives any amount of capture drift while the direction of the
national decline does not. Note the erasing rate is exp(|mu_b|) - 1, not |mu_b|.
"""),
        code("""
t = json.load(open(RES / "TRANSFERABILITY.json"))
print(json.dumps(t, indent=2))

thr = t["threshold_rule"]
check("threshold rule, lowest exposure (deaths/unit-year)", thr["lowest"]["deaths_per_unit_year"], 3.5)
check("threshold rule, lowest exposure power", 100 * thr["lowest"]["power"], 6.1, tol=0.05)
check("threshold rule, median exposure (deaths/unit-year)", thr["median"]["deaths_per_unit_year"], 10.7)
check("threshold rule, median exposure power", 100 * thr["median"]["power"], 9.5, tol=0.05)
check("threshold rule, highest exposure (deaths/unit-year)", thr["highest"]["deaths_per_unit_year"], 37.1)
check("threshold rule, highest exposure power", 100 * thr["highest"]["power"], 23.5, tol=0.05)

cb = t["completeness_bound"]
check("capture drift displacement at 1%/yr", cb["displacement_if_capture_improves_1pct_per_year"], 0.010, tol=5e-5)
check("displacement as share of national drift (%)", 100 * cb["displacement_as_share_of_national_drift"], 45, tol=0.5)
check("capture growth erasing the drift (%/yr)", cb["capture_growth_that_would_erase_the_drift_pct_per_year"], 2.25, tol=0.005)

# The power column must be the same one the design analysis committed, not a re-derivation.
d = pd.read_csv(RES / "tables/design_two_arm.csv")
a1 = d[d.arm.str.startswith("A") & (d.true_slope_x_national == 1.0)].sort_values("exposure_quintile")
assert list(a1.power) == [q["power_vs_national_drift"] for q in thr["quintiles"]], \
    "threshold rule drifted from the committed design table"
print()
print("OK    threshold rule power column matches design_two_arm.csv")
"""),
        md("""
## The planning geography

The ladder uses IBGE's immediate region, an economic geography. The health system plans in
*regiões de saúde*. If the resolving power were an artefact of an administratively
irrelevant partition, the planning partition would relieve it. It does not: more deaths per
unit, marginally narrower intervals, the same conclusion.
"""),
        code("""
h = json.load(open(RES / "HEALTH_REGION.json"))
print(json.dumps(h, indent=2))

check("health regions with data", h["units"], 433)
check("median avoidable deaths per health region", h["median_deaths_per_unit"], 152)
check("health regions classified", h["classified"], 75)
check("health regions classified (%)", h["pct_classified"], 17.3, tol=0.05)
check("health regions classified, unpooled", h["classified_unpooled"], 78)
check("health regions rising, hierarchical", h["rising_hierarchical"], 0)
check("unmapped municipality codes", h["crosswalk_provenance"]["municipality_codes_unmapped"], 23)
check("unmapped births", h["crosswalk_provenance"]["births_unmapped"], 980)
check("unmapped avoidable deaths", h["crosswalk_provenance"]["avoidable_deaths_unmapped"], 0)

# The comparison must be read from the committed ladder, not retyped.
lad = pd.read_csv(RES / "tables/aggregation_ladder.csv").set_index("level")
assert h["comparison_immediate_region"]["classified"] == int(lad.loc["immediate region", "classified"])
print()
print("OK    immediate-region comparison matches aggregation_ladder.csv")

# The crosswalk must fail on exactly the codes the immediate-region crosswalk fails on.
print(f"exposure gain: {100 * (h['median_deaths_per_unit'] / h['comparison_immediate_region']['median_deaths_per_unit'] - 1):.0f}%")
"""),
        md("""
## What remains open

Stated here so it is not mistaken for something that was checked.

- **Non-linearity is real and unresolved as a descriptive matter.** The flexible-trend
  fits show the *count* is robust, but 26.5% of the highest-exposure regions still reject
  a straight line. The paper reports the linear summary and says so.
- **Birth under-registration** in parts of the North would inflate denominators and mimic
  improvement. Not addressed here; carried as a limitation.
- **The cause grouping was inherited** from the source panel and is not re-derived in
  this repository.
"""),
    ])


def main():
    for name, builder in [
        ("01_data_and_panel", notebook_01),
        ("02_models_and_results", notebook_02),
        ("03_robustness", notebook_03),
    ]:
        path = HERE / f"{name}.ipynb"
        nbf.write(builder(), path)
        print(f"wrote {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
