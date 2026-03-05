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
example_helpers/          # Shared helper utilities for examples
examples/                 # Descriptor files + moved-project pointers
tests/                    # Automated tests
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
- Legacy imports still work for existing notebooks
