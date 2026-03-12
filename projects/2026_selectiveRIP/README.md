# 2026_selectiveRIP

SelectiveRIP project workspace restored under `projects/` for module development
and validation inside `qubit-experiment`.

## Purpose
- Keep project-specific notebooks, configs, and helper code in one place.
- Keep the reusable package and the active project workspace in the same checkout.

## Repository Layout

```text
2026_selectiveRIP/
├── docs/                 # Project-specific notebook/import conventions
├── notebooks/            # Analysis and visualization notebooks
├── noteforTG/            # Shared notes / transfer notebooks
├── custom_qubit_experiment/
│   ├── custom_experiments/
│   └── custom_analysis/
├── configs/
│   └── descriptors/      # YAML descriptor/config files
├── scripts/              # Repeatable command-line helpers
├── experiment_store/     # Local experiment outputs/checkpoints
├── qpu_parameters/       # Tracked experiment parameter snapshots
└── data/                 # Local raw data (git-ignored)
```

## Environment Setup
1. Create and activate a virtual environment from the repository root.
2. Install the editable package and development tools.
3. Open notebooks from this workspace once the root package is available.

Editable install from the `qubit-experiment/` repository root:

```bash
python -m pip install -e .[dev]
```

If this project is split out again later, the notebook bootstrap also supports a
sibling `../qubit-experiment` checkout.

## Import Guidance
- Preferred import path: `qubit_experiment.*`
- Project-local import path: `custom_qubit_experiment.*`
- Compatibility path (legacy notebooks): `experiments.*`, `analysis.*`
- Notebook bootstrap standard:
  [`docs/notebook-import-standard.md`](/Users/yalgaeahn/Research/20_Projects/qubit-experiment/projects/2026_selectiveRIP/docs/notebook-import-standard.md)

## Project-Local Custom Modules

This repo now has a project-local custom workflow area:

```text
custom_qubit_experiment/
├── custom_experiments/
└── custom_analysis/
```

Python imports use the package directly:

```python
from custom_qubit_experiment.custom_experiments import project_ramsey
from custom_qubit_experiment.custom_analysis import (
    project_ramsey as project_ramsey_analysis,
)
```

From notebooks under `notebooks/` or `noteforTG/`, initialize imports with:

```python
%run ../scripts/bootstrap_notebook.py
```

That bootstrap adds:

- this project root for `custom_qubit_experiment.*`
- the surrounding `qubit-experiment` repo root when this workspace lives under
  `projects/`
- a sibling `qubit-experiment` checkout if the project is split again later

See
[`docs/notebook-import-standard.md`](/Users/yalgaeahn/Research/20_Projects/qubit-experiment/projects/2026_selectiveRIP/docs/notebook-import-standard.md)
for the full notebook convention, required first-cell pattern, and anti-patterns.

## Data Policy
- `data/` is intentionally ignored for local/large raw data.
- `qpu_parameters/` is tracked for experiment reproducibility snapshots.

## Provenance
- Source repository: `qubit-experiment`
- Restored under `projects/` to keep project assets and reusable module tests in
  one checkout
- Split baseline tag: `v0.1.0-package-foundation`
- Split baseline commit: `a4496b2`
