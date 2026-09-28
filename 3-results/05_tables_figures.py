"""Build the manuscript tables and figures from the saved artefacts only.

Nothing is recomputed from raw data and nothing is hand-entered: every value comes from
the panel, the two canonical posteriors, or the spatial robustness posteriors. The script
also writes NUMBERS.md, which lists every figure quoted in the manuscript next to the
artefact it came from, so each one can be checked line by line.

Run:
    .venv/bin/python 3-results/05_tables_figures.py
"""

import json
import sys
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.outcome import OUTCOME, LABEL, TAG, model_dir, results_dir  # noqa: E402
PROC = ROOT / "data" / "processed"
MODEL = model_dir(ROOT)
OUT = results_dir(ROOT)
TABLES = OUT / "tables"
FIGS = OUT / "figures"

CAUSES = ["prenatal", "delivery", "newborn", "malformation", "illdef"]
LABELS = {
    "prenatal": "Antenatal care",
    "delivery": "Delivery care",
    "newborn": "Newborn care",
    "malformation": "Malformations",
    "illdef": "Ill-defined",
}

CALIBRI_DIR = Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts")
CALIBRI_FILES = [
    CALIBRI_DIR / "Calibri.ttf",
    CALIBRI_DIR / "Calibrib.ttf",
    CALIBRI_DIR / "Calibrii.ttf",
    CALIBRI_DIR / "Calibriz.ttf",
]
for font_path in CALIBRI_FILES:
    if not font_path.exists():
        raise FileNotFoundError(f"Required PPE figure font not found: {font_path}")
    font_manager.fontManager.addfont(font_path)

resolved_font = Path(
    font_manager.findfont("Calibri", fallback_to_default=False)
).resolve()
if resolved_font.name.lower() != "calibri.ttf":
    raise RuntimeError(f"Calibri regular did not resolve correctly: {resolved_font}")

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "Calibri",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def table1_data(panel):
    """Study data: deaths by action group, and the exposure trend by year."""
    total = panel[CAUSES].sum().sum()
    rows = [
        {
            "Action group": LABELS[c],
            "Deaths": int(panel[c].sum()),
            "% of deaths": round(100 * panel[c].sum() / total, 1),
        }
        for c in CAUSES
    ]
    rows.append(
        {"Action group": "All causes", "Deaths": int(total), "% of deaths": 100.0}
    )
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "table1_deaths_by_group.csv", index=False)

    by_year = (
        panel.groupby("year")
        .agg(
            Births=("births", "sum"),
            Avoidable=("avoidable", "sum"),
            Deaths=("deaths_total", "sum"),
        )
        .reset_index()
    )
    by_year["Avoidable per 1000 births"] = (
        1000 * by_year.Avoidable / by_year.Births
    ).round(2)
    by_year["NMR per 1000"] = (1000 * by_year.Deaths / by_year.Births).round(2)
    by_year.to_csv(TABLES / "table1b_exposure_by_year.csv", index=False)
    return t, by_year


def table2_parameters():
    """Posterior summaries and convergence for the two canonical models."""
    keep = [
        "mu_b",
        "sigma_b_state",
        "sigma_b_region",
        "between_state_share",
    ]
    frames = []
    for kind, extra in (("rate", "alpha"), ("share", "kappa")):
        idata = az.from_netcdf(MODEL / f"idata_{kind}.nc")
        s = az.summary(idata, var_names=keep + [extra], ci_prob=0.95)
        s = s[["mean", "eti95_lb", "eti95_ub", "ess_bulk", "r_hat"]].reset_index()
        s.insert(0, "Model", kind)
        s["divergences"] = int(idata.sample_stats.diverging.sum())
        frames.append(s)
    t = pd.concat(frames, ignore_index=True)
    t.to_csv(TABLES / "table2_parameters.csv", index=False)
    return t


def table3_robustness():
    """Classification under the three slope priors."""
    t = pd.read_csv(TABLES / "spatial_robustness.csv")
    t["share"] = (100 * t["share"]).round(1)
    t = t.rename(
        columns={
            "prior": "Slope prior",
            "classified": "Regions classified",
            "share": "% of 510",
            "median_sd": "Median posterior SD",
            "median_mde": "Median minimum detectable slope",
        }
    )
    t.to_csv(TABLES / "table3_robustness.csv", index=False)
    return t


def figure1_rope():
    """Conclusiveness as a function of the tolerance, for both estimands."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    if TAG:
        # All-cause primary: panel A is this run's rate; panel B is the avoidable rate from
        # the avoidable run, the secondary outcome the main text reports. Each panel's
        # tolerance is a multiple of its own national slope.
        panels = (
            ("A. All-cause neonatal mortality rate", TABLES / "rope_curve_rate.csv"),
            ("B. Avoidable neonatal mortality rate",
             ROOT / "3-results" / "tables" / "rope_curve_rate.csv"),
        )
    else:
        panels = (
            ("A. Avoidable neonatal mortality rate", TABLES / "rope_curve_rate.csv"),
            ("B. Avoidable share", TABLES / "rope_curve_share.csv"),
        )
    for ax, (title, path) in zip(axes, panels):
        c = pd.read_csv(path)
        x = c.tolerance_x_national
        ax.plot(
            x,
            100 * c.drifting / 510,
            color="#0072B2",
            marker="o",
            linestyle="-",
            lw=2.0,
            ms=5.5,
            label="Changing",
        )
        ax.plot(
            x,
            100 * c.credibly_flat / 510,
            color="#D55E00",
            marker="s",
            linestyle="--",
            lw=2.0,
            ms=5.5,
            label="Stable",
        )
        ax.plot(
            x,
            100 * c.indeterminate_share,
            color="0.15",
            marker="^",
            linestyle="-.",
            lw=2.2,
            ms=5.5,
            label="Indeterminate",
        )
        ax.axvline(
            1.0,
            color="0.45",
            linestyle=":",
            lw=1.5,
            label="1 × slope magnitude",
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(-2, 102)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks(x)
        tick_labels = ax.set_xticklabels([f"{value:g}" for value in x])
        tick_labels[0].set_horizontalalignment("right")
        tick_labels[1].set_horizontalalignment("left")
    axes[0].set_ylabel("% of the 510 immediate regions")
    fig.supxlabel("Tolerance (multiples of national slope magnitude)", y=0.14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        handlelength=2.0,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 1), w_pad=1.2)
    fig.savefig(
        FIGS / "figure3_tolerance_curve.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def figure2_precision(panel):
    """Why so little resolves: precision is set by the number of deaths in the region."""
    slopes = pd.read_csv(MODEL / "slopes_rate.csv")
    pooled = (
        panel.groupby("rgi_id")
        .agg(deaths=(OUTCOME, "sum"))
        .reset_index()
        .merge(slopes, on="rgi_id")
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))

    ax = axes[0]
    ok = pooled.classified_95
    ax.scatter(
        pooled.deaths[~ok], pooled.b_sd[~ok], s=7, alpha=0.45, color="0.6",
        label="Indeterminate",
    )
    ax.scatter(
        pooled.deaths[ok], pooled.b_sd[ok], s=7, alpha=0.8, color="#b2182b",
        label="Classified",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{LABEL[0].upper() + LABEL[1:]} in the region, 2014-2024")
    ax.set_ylabel("Posterior SD of the slope")
    ax.legend(frameon=False)

    ax = axes[1]
    q = pd.qcut(pooled.deaths, 5, labels=False)
    grouped = pooled.groupby(q).agg(
        rate=("classified_95", "mean"), lo=("deaths", "min"), hi=("deaths", "max")
    )
    ax.bar(
        range(len(grouped)),
        100 * grouped.rate,
        color="#4393c3",
        width=0.65,
    )
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(
        [f"{int(r.lo)}-\n{int(r.hi)}" for _, r in grouped.iterrows()]
    )
    ax.set_xlabel(f"{LABEL[0].upper() + LABEL[1:]} in the region (quintile)")
    ax.set_ylabel("% classified")
    fig.tight_layout()
    fig.savefig(FIGS / "figure2_precision.png", bbox_inches="tight")
    plt.close(fig)
    return grouped


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PROC / "panel_region_year.csv")

    t1, by_year = table1_data(panel)
    t2 = table2_parameters()
    t3 = table3_robustness()
    figure1_rope()
    figure2_precision(panel)

    rate = pd.read_csv(MODEL / "slopes_rate.csv")
    share = pd.read_csv(MODEL / "slopes_share.csv")
    # The BH flag is written by 03_precision_report.py to tables/bh_<kind>.csv; the
    # avoidable slopes file also carries it from an earlier run, the all-cause one does not.
    for name, frame in (("rate", rate), ("share", share)):
        if "bh_fdr05" not in frame.columns:
            bh = pd.read_csv(TABLES / f"bh_{name}.csv")[["rgi_id", "bh_fdr05"]]
            frame["bh_fdr05"] = frame.merge(bh, on="rgi_id", how="left").bh_fdr05.fillna(False).values
    rope_rate = pd.read_csv(TABLES / "rope_curve_rate.csv")

    numbers = {
        "regions": int(panel.rgi_id.nunique()),
        "states": int(panel.UF.nunique()),
        "panel_rows": int(len(panel)),
        "years": [int(panel.year.min()), int(panel.year.max())],
        "deaths_total": int(panel.deaths_total.sum()),
        "deaths_four_group": int(panel[CAUSES[:4]].sum().sum()),
        "deaths_avoidable": int(panel.avoidable.sum()),
        "births": int(panel.births.sum()),
        "births_first_year": int(by_year.Births.iloc[0]),
        "births_last_year": int(by_year.Births.iloc[-1]),
        "births_change_pct": round(
            100 * (by_year.Births.iloc[-1] / by_year.Births.iloc[0] - 1), 1
        ),
        "nmr_first": float(by_year["NMR per 1000"].iloc[0]),
        "nmr_last": float(by_year["NMR per 1000"].iloc[-1]),
        "classified_rate": int(rate.classified_95.sum()),
        "classified_rate_pct": round(100 * rate.classified_95.mean(), 1),
        "classified_share": int(share.classified_95.sum()),
        "classified_share_pct": round(100 * share.classified_95.mean(), 1),
        "rate_all_falling": int((rate[rate.classified_95].b_mean < 0).sum()),
        "rate_any_rising": int((rate[rate.classified_95].b_mean > 0).sum()),
        "median_mde_rate": round(float(np.median(rate.mde)), 4),
        "median_mde_share": round(float(np.median(share.mde)), 4),
        "naive_sign_error_rate": round(
            100 * float((1 - rate.prob_direction).mean()), 1
        ),
        "naive_sign_error_share": round(
            100 * float((1 - share.prob_direction).mean()), 1
        ),
        "bh_rate": int(rate.bh_fdr05.sum()),
        "bh_share": int(share.bh_fdr05.sum()),
        "rope_at_1x_drifting": int(
            rope_rate.loc[rope_rate.tolerance_x_national == 1.0, "drifting"].iloc[0]
        ),
        "rope_at_1x_flat": int(
            rope_rate.loc[rope_rate.tolerance_x_national == 1.0, "credibly_flat"].iloc[0]
        ),
        "rope_at_1x_indeterminate_pct": round(
            100
            * float(
                rope_rate.loc[
                    rope_rate.tolerance_x_national == 1.0, "indeterminate_share"
                ].iloc[0]
            ),
            1,
        ),
        "joint_both": int((rate.classified_95 & share.classified_95).sum()),
        "joint_rate_only": int((rate.classified_95 & ~share.classified_95).sum()),
        "joint_share_only": int((~rate.classified_95 & share.classified_95).sum()),
        "adjacency": (PROC / "adjacency_report.txt").read_text().strip().split("\n"),
        "robustness": t3.to_dict("records"),
        "parameters": t2.to_dict("records"),
    }
    (OUT / "NUMBERS.json").write_text(json.dumps(numbers, indent=2))

    lines = ["# Every number quoted in the manuscript", ""]
    for k, v in numbers.items():
        if isinstance(v, (list, dict)):
            lines.append(f"- **{k}**:")
            lines.append("")
            lines.append("```")
            lines.append(json.dumps(v, indent=2))
            lines.append("```")
        else:
            lines.append(f"- **{k}**: {v}")
    (OUT / "NUMBERS.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({k: v for k, v in numbers.items() if not isinstance(v, (list, dict))}, indent=2))
    print("\nwrote tables, figures, NUMBERS.json and NUMBERS.md")


if __name__ == "__main__":
    main()
