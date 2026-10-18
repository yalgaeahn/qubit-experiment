# scripts

Repeatable helpers for this project.

- `bootstrap_notebook.py`: add this project root and the enclosing
  `qubit-experiment` repo root to `sys.path` from a notebook via
  `%run ../scripts/bootstrap_notebook.py`, then import project-local modules from
  `custom_qubit_experiment.*`
- If the project is split into a separate checkout again later, the same bootstrap
  also falls back to a sibling `qubit-experiment` repo when present
- The notebook-side usage contract is documented in
  [`../docs/notebook-import-standard.md`](/Users/yalgaeahn/Research/20_Projects/qubit-experiment/projects/2026_selectiveRIP/docs/notebook-import-standard.md)
- future data sync, report generation, and batch post-processing helpers should
  also live here
