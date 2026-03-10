# qubit-experiment

Reusable LabOne Q experiment/analysis package focused on reusable code.
Project-specific notebooks and assets are maintained in separate project repositories.

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

Use `qubit_experiment` imports:

```python
from qubit_experiment.analysis.resonator_spectroscopy import analysis_workflow
from qubit_experiment.experiments.qubit_spectroscopy import experiment_workflow
from qubit_experiment.helper import load_qubit_parameters
```

## Repository Layout

```text
qubit_experiment/         # Canonical reusable package source
example_helpers/          # Shared helper utilities for examples
examples/                 # Descriptor files + moved-project pointers
tests/                    # Automated tests
projects/                 # Local-only scratch workspaces for real-data development
```

## Examples Split

- `examples/selectiveRIP` now contains a pointer only.
- Full selectiveRIP notebooks/configs/scripts are in:
  - https://github.com/yalgaeahn/2026_selectiveRIP
- Split baseline:
  - tag `v0.1.0-package-foundation`
  - commit `a4496b2`

## Project Repository Policy

- Keep reusable workflows and helpers in `qubit_experiment/`.
- Keep project/lab-specific notebooks, calibration data, and exploratory scripts in
  separate project repositories.
- For temporary local validation inside this repo, use `projects/<workspace>/`; this
  area is ignored so private data and scratch notebooks stay out of the public repo.
- Use those project repositories with `qubit-experiment` as a dependency (editable or
  Git URL install).

## Development Checks

```bash
ruff check . --fix
black .
mypy qubit_experiment
pytest -q
```

## Publication Readiness Checklist

- Reusable logic is under `qubit_experiment/`
- Project/lab specific assets live in separate project repositories
- New code uses `qubit_experiment.*` imports
- No top-level compatibility packages are required
