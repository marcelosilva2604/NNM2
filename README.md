# How much of Brazil's neonatal transition can the death registry actually resolve?

Analysis code for a study of whether routine vital registration can detect change in
avoidable neonatal mortality at the spatial scale where care is organised, Brazil
2014-2024.

The paper asks a question borrowed from hospital quality profiling and rarely put to a
national civil registration system: not *how many deaths were there*, but *how much
change could this registry have detected at all*. The answer is computed at four nested
spatial scales plus the SUS health-planning geography, and reported as a design analysis
(power, type S and type M errors) rather than as a list of significant places.

The manuscript is not yet published. This repository holds the code and the derived
results; it does not hold the manuscript text.

---

## What is here, and what is not

**Here.** Every script that builds the panels, fits the models and produces the reported
quantities; the derived result tables and JSON summaries those scripts wrote; the three
audit notebooks, **committed with their outputs**; and the two municipality crosswalks.

**Not here, and why.**

| Missing | Why | How to get it |
|---|---|---|
| `data/raw/`, `data/processed/` | The source records are public and it is better to fetch them than to mirror them | SIM and SINASC via DATASUS, https://datasus.saude.gov.br |
| `2-model/*.nc` (5.2 GB of posteriors) | Single files reach 1 GB, against GitHub's 100 MB limit | Re-run the scripts in `2-model/`; each carries a fixed seed |
| `data/ref/*.geojson` | Boundary files, 28 MB, and freely available | IBGE |
| Manuscript, supplement, cover letter | Not published until the paper is | |

The notebooks are versioned **executed** on purpose. Since the posteriors are not here, a
cleared notebook would prove nothing; with outputs, you can read every assertion and its
result on GitHub without running anything. `notebooks/build_notebooks.py` generates them, so
what you see is exactly what that file says.

---

## Layout

```
1-data/      panel construction, contiguity graph, covariates
2-model/     every fit; each writes a posterior (.nc, not versioned) and a .log
3-results/   result scripts, plus the tables and JSON summaries they produced
notebooks/   the audit notebooks: they assert, they do not fit
data/ref/    municipality crosswalks and their provenance
```

## Reproducing

Fitting the full set takes on the order of a day on a laptop. Model scripts are resumable:
a fit already present as `2-model/idata_*.nc` is loaded rather than re-sampled, so
re-running is cheap and deleting one file forces a genuine refit.

| Step | Script | Produces |
|---|---|---|
| 0 | `1-data/00_cause_grouping.py` | ICD-10 to action-group rule table, observed-code list, municipality-year cause panel (asserted identical to the one used) |
| 1 | `1-data/01_build_panel.py` | region-year and municipality-year panels, build report |
| 2 | `1-data/02_build_adjacency.py` | contiguity graph for the spatial prior |
| 3 | `1-data/03_build_covariates.py` | region-year covariates |
| 4 | `2-model/02_canonical_fit.py` | **canonical** rate and share posteriors |
| 5 | `2-model/07_state_model.py` | state-level fit |
| 6 | `2-model/09_aggregation_ladder.py` | four nested partitions |
| 7 | `2-model/04_spatial_robustness.py` | three slope priors |
| 8 | `2-model/08_ppc_and_dispersion.py` | predictive checks, exposure-dependent dispersion |
| 9 | `2-model/12_seed_stability.py` | eight refits under different seeds |
| 10 | `2-model/13_flexible_trend.py` | quadratic and random-walk trend shapes |
| 11 | `2-model/14_covariates.py` | covariate-assisted specifications |
| 12 | `2-model/15_state_dispersion.py` | state fits under two variance functions, unpooled contrasts |
| 13 | `2-model/19_health_region.py` | the SUS planning geography, as a competing partition |
| 14 | `3-results/16_design_two_arm.py` | design analysis, both arms |
| 15 | `3-results/17_orphan_numbers.py` | quantities no other script produced |
| 16 | `3-results/18_transferability.py` | threshold rule and completeness bound |
| 17 | `3-results/03,05,06,07,11_*.py` | tables, figures, precision, supporting checks |

`3-results/10_design_analysis.py` is superseded by step 14 and no longer feeds the paper; it
is kept only so the earlier single-arm result can be reproduced.

**The rebuild path, and what is still upstream.** Step 0
(`1-data/00_cause_grouping.py`) now derives the cause panel
(`muni_year_cause_panel.csv`) from the SIM extract in `data/raw`: it applies the ICD-10 to
action-group mapping documented in `data/ref/ICD10_ACTION_GROUPS.md`, drops the 196
records whose residence code is a state-level "municipality unknown" code, and asserts
that the rebuilt panel is identical to the one the analysis used. Three other
intermediates consumed by step 1 (`nascidos_muni_ano.csv`, `cnes_muni_ano.csv`,
`idhm_muni.csv`) are still produced by an earlier pipeline outside this repository; they
are aggregations of public SINASC, CNES and Atlas Brasil tables by municipality and year.

## Checking without refitting

```bash
python notebooks/run_notebooks.py
```

The three notebooks load the stored posteriors and derived tables, recompute the study's
load-bearing quantities, and **assert** each against the published value. They refit
nothing. A failure means the text and the artefacts have diverged, which is the intended
alarm. As committed, all three execute cleanly, and their committed outputs show it.

Coverage is deliberate but not total: the assertions cover the panel totals, the
aggregation ladder, the health-region comparison, the departure contrasts, the design
analysis, the tolerance curve, the prior and trend-shape robustness, the seed spread, the
predictive-check gradient, the covariate counts, the transferability rule, the completeness
bound and the supporting checks. Numbers appearing only in prose, such as individual
interval endpoints, are not asserted.

Running them requires the posteriors, so it is something you can do after a refit, not on a
fresh clone. Reading the committed outputs requires nothing.

---

## Conventions that are enforced, not merely intended

- **All intervals are 95% equal-tailed posterior intervals.** No other width is used
  anywhere. An earlier version of this project mixed 94%, 95% and a z cutoff across
  scripts, which is why the same count appeared as 85, 87, 88 and 95 in different files.
- **A unit is "classified" when its 95% interval excludes zero.** One rule, one
  implementation.
- **Unpooled fits use quasi-Poisson referred to t on n-2 residual degrees of freedom.**
  With 11 annual points the normal critical value of 1.96 runs at a 7-8% false positive
  rate. An earlier version used 1.96 and inflated every unpooled figure by roughly half.
- **Posterior predictive Pearson residuals are divided by residual degrees of freedom**,
  not by the number of observations. The wrong divisor makes a correctly specified model
  look overdispersed.
- **Sampling happens only in scripts.** Notebooks and result scripts read `.nc` files.

---

## Things that were found wrong and corrected

Recorded because a reader should be able to see what was repaired rather than have to
rediscover it.

1. The panel keyed on deaths rather than births, dropping 21,128 zero-death
   municipality-years (34%), concentrated in the smallest units. Rebuilt on the birth
   series.
2. Between-state variance was computed as an unadjusted eta-squared on posterior means,
   whose null expectation is 5%. Replaced by a three-level model parameter.
3. Per-state magnitudes were quoted from shrunken posterior means; three of six reverse
   sign against the state's own series. All per-state magnitudes removed.
4. A claimed year-to-year variability of "about 13%" in the largest states was
   `1/sqrt(alpha)`, a model assumption, not data (the observed figures are 4.3% and
   4.1%). Removed.
5. The posterior predictive block never ran: it extended a temporary thinned copy of the
   InferenceData and saved the original. Fixed in `08_ppc_and_dispersion.py`.
6. The unpooled critical value bug described above.
7. `counts.ptp()` (removed in numpy 2) crashed the seed-stability summary after the
   fits had completed.
8. The state model kept a single dispersion after the same defect had been repaired at
   region level. It imposed a common floor on relative variability regardless of size,
   inflating the largest states' slope uncertainty two- to threefold (the noise floor it
   imposes is about 13% against an observed 4%), and under it no state separated from the
   national drift. With exposure-dependent dispersion six do, and unpooled ten do. A
   published conclusion changed.
9. `09_aggregation_ladder.py` did not call `unpooled_ladder()`, so the committed table
   carried columns the documented pipeline could not regenerate.
10. One notebook assertion compared a published constant to itself.
11. The completeness bound was typed in as 2.23% per year, the linear reading of the
    national drift. Capture that grows geometrically displaces the fitted slope by
    log(1 + r), so the rate that erases the drift is exp(|mu_b|) - 1 = 2.25%. Corrected,
    and the calculation moved into `18_transferability.py`.

---

## What this project does not establish

- The **ICD-10 to action-group mapping** is published (`data/ref/ICD10_ACTION_GROUPS.md`,
  `icd10_action_group_rules.csv`, `icd10_codes_observed_2014_2024.csv`) and re-applied by
  step 0, but it assigns avoidability by cause code, following the Brazilian avoidable-causes
  list, and was not validated against medical records.
- **Birth under-registration**, concentrated in parts of the North, would inflate
  denominators and mimic improvement. Not addressed.
- **Death-count completeness** is a larger threat than the birth denominator. It is bounded
  arithmetically in the paper, not measured: capture improving at 1% a year displaces a
  slope by +0.010 per year, 45% of the national drift, and 2.25% a year would erase that
  drift. Because the displacement is additive in the mean, it moves where a slope sits
  without changing how precisely it is estimated, so the detectability result survives it
  while the direction of the national decline does not.
- **The health-region partition is a December 2024 vintage applied retrospectively.** The
  study fits the SUS planning geography as well as the IBGE immediate region (step 13), and
  the two agree. But health regions were repactuated during the decade in several states,
  so the unit is held fixed at a definition that did not hold throughout. This is
  unavoidable for a within-unit trend.
- **Terminal-year maturity.** The SIM and SINASC files for 2014-2024 were the Ministry of
  Health's final releases at the time of download (the Ministry had moved on to the 2025
  files). An under-complete terminal year would bias every slope toward apparent
  improvement, which is why this is recorded here; no completeness flag is carried in the
  extract itself.
- **Non-linearity** is real: a region's own series rejects the log-linear fit in 13.3% of
  cases overall and 26.5% in the highest-exposure quintile. The classification count is
  robust to trend shape, but the linear summary remains a simplification where the study
  can actually resolve.
- The **macro-region rung** of the ladder is uninterpretable at five units and is reported
  as such, not as a finding.
- The **municipal boundary file** carries no provenance metadata.

---

## Data sources

SIM (mortality) and SINASC (live births) via DATASUS, 2014-2024, by municipality of
residence; CNES for establishment counts; IBGE crosswalks and municipal boundaries; Atlas
Brasil 2010 for the development index; and the DATASUS territorial base (December 2024) for
the municipality-to-health-region crosswalk, whose provenance is documented in
`data/ref/muni_regiao_saude_PROVENANCE.md`.

The SIM extract is publicly available, de-identified mortality microdata (one record per
death, no direct personal identifiers); SINASC, CNES and Atlas Brasil enter as aggregate
counts by municipality and year. Every result is aggregated at municipality-year level or
coarser.

---

## Licence and citation

Code is released under the MIT Licence (see `LICENSE`). The derived result tables and
JSON summaries in `3-results/` and the reference tables in `data/ref/` may be reused
with attribution to this repository and to the paper once it is published. The source
records belong to the Brazilian Ministry of Health (SIM, SINASC, CNES via DATASUS) and
to IBGE, under their own terms.
