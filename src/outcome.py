"""Outcome selection for the whole pipeline.

The manuscript submitted to EJE and PPE used avoidable neonatal deaths (Brazilian list of
avoidable causes) as the primary outcome. PPE's editorial office rejected it on 27 Sep 2026
because that outcome "is not commonly used in the perinatal or neonatal mortality
literature". This module lets every model and result script run unchanged on all-cause
neonatal deaths instead, so the two outcomes can be compared without either overwriting
the other.

Set the environment variable NNM2_OUTCOME before running any script:

    avoidable      avoidable neonatal deaths (default; reproduces the submitted results)
    deaths_total   all-cause neonatal deaths

The default writes to 2-model/ and 3-results/ exactly as before. Any other outcome writes
to 2-model/<tag>/ and 3-results/<tag>/ (tables and figures included), so the original
posteriors, tables and JSON summaries are never touched.

The avoidable share outcome (avoidable deaths out of the four action groups) does not
depend on this switch: it is the same beta-binomial model under both settings.
"""

import os
from pathlib import Path

OUTCOMES = {
    "avoidable": {"tag": "", "label": "avoidable neonatal deaths"},
    "deaths_total": {"tag": "allcause", "label": "all-cause neonatal deaths"},
}

OUTCOME = os.environ.get("NNM2_OUTCOME", "avoidable")
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
