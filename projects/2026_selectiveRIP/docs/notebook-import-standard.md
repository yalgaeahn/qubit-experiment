# Notebook Import Standard

This document defines the import/bootstrap convention for notebooks in
`projects/2026_selectiveRIP/`.

It applies to notebooks under both:

- `notebooks/`
- `noteforTG/`

## Goal

Every notebook in this project should be able to import both:

- reusable package modules from `qubit_experiment.*`
- project-local modules from `custom_qubit_experiment.*`

without hand-written `sys.path` logic in the notebook itself.

## Required First Cell

Start the first code cell with:

```python
%run ../scripts/bootstrap_notebook.py
```

Then use the exported `PROJECT_ROOT` variable for project-relative paths:

```python
%run ../scripts/bootstrap_notebook.py

from pathlib import Path
from laboneq.serializers import load

repo_root = PROJECT_ROOT
descriptor_path = repo_root / "configs" / "descriptors" / "1port.yaml"
```

## Canonical Import Paths

Use these import roots in notebooks:

- `qubit_experiment.*` for reusable package code
- `custom_qubit_experiment.*` for project-local custom workflows/analysis

Legacy compatibility imports such as `experiments.*` or `analysis.*` should not
be used in new notebook edits.

## What The Bootstrap Adds

`bootstrap_notebook.py` adds the following to `sys.path`:

- the project root, so `custom_qubit_experiment.*` is importable
- the enclosing `qubit-experiment` repository root when this project lives under
  `projects/`
- a sibling `../qubit-experiment` checkout when the project is split out into a
  separate directory later

This keeps notebooks working in both monorepo and split-checkout layouts.

## Do Not Do This In Notebooks

Avoid:

- manual `sys.path.insert(...)` calls
- scanning `Path.cwd()` parents to guess the repo root
- inferring the correct root from the presence of folders such as `configs/`,
  `notebooks/`, or `qpu_parameters/`

Those patterns are fragile because this project contains both a project root and
a surrounding package root, and notebooks often need both.

## Practical Rule

1. Bootstrap with `%run ../scripts/bootstrap_notebook.py`.
2. Set `repo_root = PROJECT_ROOT`.
3. Import from `qubit_experiment.*` and `custom_qubit_experiment.*`.
4. Build all project-relative file paths from `repo_root`.

## References

- [`../scripts/bootstrap_notebook.py`](/Users/yalgaeahn/Research/20_Projects/qubit-experiment/projects/2026_selectiveRIP/scripts/bootstrap_notebook.py)
- [`../README.md`](/Users/yalgaeahn/Research/20_Projects/qubit-experiment/projects/2026_selectiveRIP/README.md)
- [`../../../tests/test_bootstrap_notebook.py`](/Users/yalgaeahn/Research/20_Projects/qubit-experiment/tests/test_bootstrap_notebook.py)
- [`../../../tests/test_readout_optimization_notebook.py`](/Users/yalgaeahn/Research/20_Projects/qubit-experiment/tests/test_readout_optimization_notebook.py)
