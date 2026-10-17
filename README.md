# Spatial scale and the detectability of neonatal mortality trends in Brazil, 2014-2024

Analysis code and derived results for a registry-based design analysis of all-cause neonatal
mortality (deaths at 0-27 completed days) in Brazil, 2014-2024, using the Mortality
Information System (SIM) and the Live Birth Information System (SINASC). The study asks how
many local trends the registry counts can resolve at four nested geographic scales (5,570
municipalities, 510 immediate regions, 27 states, five macro-regions), under no pooling,
hierarchical pooling, and a scaled BYM2 spatial prior, and evaluates power and wrong-sign
(type S) and magnitude (type M) errors in matched simulations.

## Layout

| Folder | Content |
|---|---|
| `1-data/` | Panel construction from the public records, adjacency graph, covariates |
| `2-model/` | Model fits (negative-binomial trend models, aggregation ladder, state model, scaled BYM2, robustness) and their slope tables in `2-model/allcause/` |
| `3-results/` | Result, design-analysis, simulation, audit and figure scripts; outputs in `3-results/allcause/` (tables, figures, JSON summaries, per-replicate simulation files in `matched/`) |
| `src/outcome.py` | Output-folder selection (all-cause neonatal deaths) |
| `data/ref/` | Municipality to immediate-region and health-region crosswalks |
| `run_allcause.sh` | Order in which the scripts were run |

## Not included

| Item | Reason | How to obtain |
|---|---|---|
| `data/raw/`, `data/processed/` | The source records are public | SIM and SINASC from DATASUS, https://datasus.saude.gov.br |
| `2-model/**/*.nc` (sampled posteriors) | Several GB, above GitHub's file limit | Re-run the scripts in `2-model/`; each uses a fixed seed |
| `data/ref/*.geojson` | Boundary files | IBGE |

## Reproducing

1. Download the SIM and SINASC records for 2014-2024 into `data/raw/` and run the three scripts in `1-data/`.
2. Create the environment with `requirements.txt` (Python, PyMC, ArviZ, geopandas).
3. Run `./run_allcause.sh`. The default outcome is all-cause neonatal deaths.

Posterior sampling is stochastic across platforms; the fixed seeds reproduce the stored
tables on the platform used for the study.

## License

MIT, see `LICENSE`.
