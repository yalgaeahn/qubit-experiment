# scripts

Repeatable helpers for this project.

- `bootstrap_notebook.py`: add this project root and the enclosing
  `qubit-experiment` repo root to `sys.path` from a notebook via
  `%run ../scripts/bootstrap_notebook.py`, then import project-local modules from
  `custom_qubit_experiment.*`
- If the project is split into a separate checkout again later, the same bootstrap
  also falls back to a sibling `qubit-experiment` repo when present
- future data sync, report generation, and batch post-processing helpers should
  also live here
