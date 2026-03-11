# qubit-experiment

Reusable LabOne Q experiment/analysis package focused on reusable code.
Project-specific notebooks and assets currently live under `projects/`, with
`projects/2026_selectiveRIP/` restored for active module development and
validation.

## Environment Setup

Requirements:

- Python 3.10+
- Access to Zurich Instruments LabOne Q packages

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Package naming note:

- Install name in `pip`: `qubit-experiment`
- Import name in Python: `qubit_experiment`

### Install

Use a regular install when you want a stable copy of the package inside the
environment and do not need source edits in this repository to take effect
immediately.

Install the package:

```bash
python -m pip install .
```

What this does:

- Builds the package from the current repository state
- Installs a copy into `.venv/lib/python*/site-packages`
- Requires re-running `pip install .` after local source changes

Useful checks:

```bash
python -m pip show qubit-experiment
python -c "import qubit_experiment; print(qubit_experiment.__file__)"
```

### Editable Install

Use an editable install for development in this repository. This is the
recommended setup when editing code, running tests repeatedly, or using local
notebooks against in-progress changes.

Install the package in editable mode with development tools:

```bash
python -m pip install -e .[dev]
```

What this does:

- Installs package metadata into `site-packages`
- Points the environment back to this working tree instead of copying files
- Makes source edits under `qubit_experiment/` visible immediately

Useful checks:

```bash
python -m pip list | rg 'qubit-experiment'
python -m pip show qubit-experiment
python -c "import qubit_experiment; print(qubit_experiment.__file__)"
```

In editable mode, it is normal not to see a full `site-packages/qubit_experiment/`
directory. You will usually see files such as:

- `__editable__.qubit_experiment-0.1.0.pth`
- `__editable___qubit_experiment_0_1_0_finder.py`
- `qubit_experiment-0.1.0.dist-info/`

### Switching Between Install Modes

Switch from editable install to regular install:

```bash
python -m pip uninstall qubit-experiment
python -m pip install .
```

Switch from regular install to editable install:

```bash
python -m pip uninstall qubit-experiment
python -m pip install -e .[dev]
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
examples/                 # Lightweight pointers and example entrypoints
tests/                    # Automated tests
projects/                 # Project workspaces; selectiveRIP lives here today
```

## SelectiveRIP Workspace

- Active selectiveRIP notebooks, configs, scripts, and project-local helpers are in
  `projects/2026_selectiveRIP/`.
- `examples/selectiveRIP/` remains a pointer README only.
- Historical split baseline:
  - tag `v0.1.0-package-foundation`
  - commit `a4496b2`

## Project Workspace Policy

- Keep reusable workflows and helpers in `qubit_experiment/`.
- Keep project/lab-specific notebooks, calibration data, exploratory scripts, and
  project-only wrappers under `projects/<workspace>/`.
- `projects/2026_selectiveRIP/` is the current shared workspace used to validate
  reusable module changes against real project assets.
- Promote anything reusable from `projects/<workspace>/` into
  `qubit_experiment/` and `tests/`.
- Keep raw data and transient outputs out of the tracked package surface.

## Development Checks

```bash
ruff check . --fix
black .
mypy qubit_experiment
pytest -q
```

## Publication Readiness Checklist

- Reusable logic is under `qubit_experiment/`
- Project/lab specific assets stay isolated under `projects/<workspace>/`
- New code uses `qubit_experiment.*` imports
- No top-level compatibility packages are required
