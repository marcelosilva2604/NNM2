# Every number quoted in the manuscript

- **regions**: 510
- **states**: 27
- **panel_rows**: 5610
- **years**:

```
[
  2014,
  2024
]
```
- **deaths_total**: 260023
- **deaths_four_group**: 245796
- **deaths_avoidable**: 117578
- **births**: 30462542
- **births_first_year**: 2979133
- **births_last_year**: 2384384
- **births_change_pct**: -20.0
- **nmr_first**: 8.89
- **nmr_last**: 8.32
- **classified_rate**: 86
- **classified_rate_pct**: 16.9
- **classified_share**: 67
- **classified_share_pct**: 13.1
- **rate_all_falling**: 86
- **rate_any_rising**: 0
- **median_mde_rate**: 0.0343
- **median_mde_share**: 0.057
- **naive_sign_error_rate**: 13.1
- **naive_sign_error_share**: 18.7
- **bh_rate**: 10
- **bh_share**: 16
- **rope_at_1x_drifting**: 9
- **rope_at_1x_flat**: 0
- **rope_at_1x_indeterminate_pct**: 98.2
- **joint_both**: 43
- **joint_rate_only**: 43
- **joint_share_only**: 24
- **adjacency**:

```
[
  "Immediate-region adjacency",
  "==================================================",
  "regions              : 510",
  "matches panel order  : True",
  "undirected pairs     : 1,324",
  "mean degree          : 5.19",
  "min / max degree     : 1 / 13",
  "islands (no queen nb): 1",
  "components welded    : 5",
  "    joined 210019 <-> 170008  (90 km apart)",
  "    joined 310012 <-> 290016  (101 km apart)",
  "    joined 410022 <-> 350017  (56 km apart)",
  "    joined 520012 <-> 310060  (75 km apart)",
  "    joined 530001 <-> 520020  (46 km apart)",
  "connected components : 1  (OK for ICAR)"
]
```
- **robustness**:

```
[
  {
    "Slope prior": "nopool",
    "Regions classified": 77,
    "% of 510": 15.1,
    "Median posterior SD": 0.0322456453155092,
    "Median minimum detectable slope": 0.0632014648183981,
    "divergences": 0
  },
  {
    "Slope prior": "exch",
    "Regions classified": 84,
    "% of 510": 16.5,
    "Median posterior SD": 0.0175507942437553,
    "Median minimum detectable slope": 0.0343995567177604,
    "divergences": 0
  },
  {
    "Slope prior": "bym2",
    "Regions classified": 89,
    "% of 510": 17.5,
    "Median posterior SD": 0.0174240721823055,
    "Median minimum detectable slope": 0.0341511814773189,
    "divergences": 0
  }
]
```
- **parameters**:

```
[
  {
    "Model": "rate",
    "index": "mu_b",
    "mean": "-0.02227",
    "eti95_lb": "-0.027",
    "eti95_ub": "-0.017",
    "ess_bulk": 3167,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "rate",
    "index": "sigma_b_state",
    "mean": "0.0078",
    "eti95_lb": "0.0012",
    "eti95_ub": "0.015",
    "ess_bulk": 889,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "rate",
    "index": "sigma_b_region",
    "mean": "0.02002",
    "eti95_lb": "0.016",
    "eti95_ub": "0.024",
    "ess_bulk": 2501,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "rate",
    "index": "between_state_share",
    "mean": "0.146",
    "eti95_lb": "0.0031",
    "eti95_ub": "0.38",
    "ess_bulk": 878,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "rate",
    "index": "alpha",
    "mean": "56.39",
    "eti95_lb": "49",
    "eti95_ub": "65",
    "ess_bulk": 7137,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "share",
    "index": "mu_b",
    "mean": "-0.0282",
    "eti95_lb": "-0.035",
    "eti95_ub": "-0.021",
    "ess_bulk": 3609,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "share",
    "index": "sigma_b_state",
    "mean": "0.0111",
    "eti95_lb": "0.0011",
    "eti95_ub": "0.023",
    "ess_bulk": 806,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "share",
    "index": "sigma_b_region",
    "mean": "0.0361",
    "eti95_lb": "0.031",
    "eti95_ub": "0.042",
    "ess_bulk": 2683,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "share",
    "index": "between_state_share",
    "mean": "0.101",
    "eti95_lb": "0.00083",
    "eti95_ub": "0.31",
    "ess_bulk": 807,
    "r_hat": "1.00",
    "divergences": 0
  },
  {
    "Model": "share",
    "index": "kappa",
    "mean": "148.7",
    "eti95_lb": "120",
    "eti95_ub": "180",
    "ess_bulk": 7174,
    "r_hat": "1.00",
    "divergences": 0
  }
]
```
