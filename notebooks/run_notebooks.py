"""Execute the notebooks in place, so the committed copies carry real output.

A notebook whose cells have never run is documentation, not evidence. Executing them
here means every assertion in them has actually passed against the artefacts on disk at
the time of the run.

Run:
    .venv/bin/python notebooks/run_notebooks.py
"""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

failed = []
for name in ("01_data_and_panel", "02_models_and_results", "03_robustness"):
    path = HERE / f"{name}.ipynb"
    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
    )
    try:
        client.execute()
        status = "OK"
    except Exception as exc:  # noqa: BLE001 - a failed assertion is a real finding
        status = f"FAILED: {type(exc).__name__}: {str(exc)[:300]}"
        failed.append(name)
    nbf.write(notebook, path)
    print(f"{name}: {status}")

# A failed assertion is the alarm this suite exists for; it must set the exit code so
# anything automated can gate on it.
if failed:
    sys.exit(1)
