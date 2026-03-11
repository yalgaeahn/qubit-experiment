# custom_qubit_experiment

Project-local custom workflow area for `2026_selectiveRIP`.

## Layout

- `custom_experiments/`: local experiment workflows and wrappers
- `custom_analysis/`: local analysis workflows and helpers

## Import names

Python imports use this package directly:

```python
from custom_qubit_experiment.custom_experiments import project_ramsey
from custom_qubit_experiment.custom_analysis import (
    project_ramsey as project_ramsey_analysis,
)
```

## Notebook bootstrap

From a notebook inside `notebooks/`, run:

```python
%run ../scripts/bootstrap_notebook.py
```

That adds this project root plus the surrounding `qubit-experiment` repo root to
`sys.path` when present. If the project is split into a separate checkout again
later, it also falls back to a sibling `qubit-experiment` repo.
