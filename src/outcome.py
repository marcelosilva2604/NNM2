"""Outcome selection for the whole pipeline.

The study analyses all-cause neonatal deaths. Every script reads the outcome from the
environment variable NNM2_OUTCOME, which defaults to deaths_total, and writes its output to
2-model/allcause/ and 3-results/allcause/.

    deaths_total   all-cause neonatal deaths (default)
"""

import os
from pathlib import Path

OUTCOMES = {
    "deaths_total": {"tag": "allcause", "label": "all-cause neonatal deaths"},
}

OUTCOME = os.environ.get("NNM2_OUTCOME", "deaths_total")
if OUTCOME not in OUTCOMES:
    raise ValueError(
        f"NNM2_OUTCOME={OUTCOME!r} is not one of {sorted(OUTCOMES)}"
    )

TAG = OUTCOMES[OUTCOME]["tag"]
LABEL = OUTCOMES[OUTCOME]["label"]


def _subdir(base: Path) -> Path:
    d = base / TAG if TAG else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir(root: Path) -> Path:
    """Where posteriors and slope tables live for the selected outcome."""
    return _subdir(root / "2-model")


def results_dir(root: Path) -> Path:
    """Where tables, figures and JSON summaries live for the selected outcome."""
    return _subdir(root / "3-results")
