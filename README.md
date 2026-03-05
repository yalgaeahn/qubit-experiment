# qubit-experiment

Reusable LabOne Q experiment/analysis package with a clean split between:

- canonical reusable package code (`qubit_experiment/...`)
- project-specific exploratory work (`projects/<project>/...`)

## Installation

Requirements:

- Python 3.10+
- Access to Zurich Instruments LabOne Q packages

Install for use:

```bash
pip install .
```

Install for development:

```bash
pip install -e .[dev]
```

## Canonical Imports

Use `qubit_experiment` imports in new code:

```python
from qubit_experiment.analysis.resonator_spectroscopy import analysis_workflow
from qubit_experiment.experiments.qubit_spectroscopy import experiment_workflow
from qubit_experiment.helper import load_qubit_parameters
```

Legacy imports remain available for compatibility:

- `analysis.*`
- `experiments.*`
- `qpu_types.*`
- `helper_functions.*`
- `helper`, `operations`, `custom_pulse_library`

## Repository Layout

```text
qubit_experiment/         # Canonical reusable package source
analysis/                 # Legacy compatibility package (for old imports)
experiments/              # Legacy compatibility package
qpu_types/                # Legacy compatibility package
helper_functions/         # Legacy compatibility package
helper.py                 # Legacy compatibility module
operations.py             # Legacy compatibility module
custom_pulse_library.py   # Legacy compatibility module
projects/                 # Project-specific workspaces (not package API)
projects/kaist/notebooks/ # KAIST-specific exploratory notebooks
examples/                 # Curated/public examples (cleanup can be done separately)
tests/                    # Automated tests
```

## Project-Specific Code Policy

Use `projects/<project>/...` for:

- ad-hoc notebooks
- local calibration scripts
- intermediate artifacts

Keep reusable workflows and helpers in `qubit_experiment/`.

## Development Checks

```bash
ruff check . --fix
black .
mypy qubit_experiment
pytest -q
```

## Publication Readiness Checklist

- Reusable logic is under `qubit_experiment/`
- Project/lab specific notebooks are under `projects/`
- New code uses `qubit_experiment.*` imports
- Legacy imports still work for existing notebooks
