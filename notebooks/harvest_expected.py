"""Harvest expected values for a new outcome from an executed notebook run.

Workflow for a new outcome: run build_notebooks.py and run_notebooks.py with the expected
JSON empty, so every check prints "SKIP <label>: no expected value yet (recomputed=X)".
This script reads those lines from the executed notebooks and writes them into the
expected JSON, after which run_notebooks.py asserts against them. The JSON is then the
record of the published numbers, and any later drift between artefacts and text fails.

Run (after run_notebooks.py, with the same NNM2_OUTCOME):
    .venv/bin/python notebooks/harvest_expected.py
"""

import json
import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.outcome import TAG  # noqa: E402

folder = HERE / TAG if TAG else HERE
target = HERE / f"expected_{TAG or 'avoidable'}.json"
pattern = re.compile(r"SKIP\s+(.+?): no expected value yet \(recomputed=(.+?)\)$")

existing = json.loads(target.read_text()) if target.exists() else {}
found = {}
for name in ("01_data_and_panel", "02_models_and_results", "03_robustness"):
    nb = nbf.read(folder / f"{name}.ipynb", as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = out.get("text", "") if out.get("output_type") == "stream" else ""
            for line in text.splitlines():
                m = pattern.match(line.strip())
                if m:
                    label, value = m.group(1), m.group(2)
                    try:
                        found[label] = json.loads(value)
                    except json.JSONDecodeError:
                        found[label] = float(value)

new = {k: v for k, v in found.items() if k not in existing}
merged = {**existing, **new}
target.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
print(f"{len(found)} SKIP values found, {len(new)} added, {len(merged)} labels in {target.name}")
