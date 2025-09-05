# qubit-experiment

Experiment and analysis tools for superconducting-qubit workflows built on LabOne Q. Includes experiments, analysis, pulse helpers, and example notebooks.

## Installation

Requirements:

- Python 3.10+
- Access to Zurich Instruments LabOne Q packages

Install (user):

```bash
pip install .
```

Install (development, editable):

```bash
pip install -e .[dev]
```

This installs optional tools for linting/formatting/type checking and notebooks.

## Development

Recommended workflow in a virtual environment (conda/venv/uv):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Quality tools:

```bash
# Lint and import order
ruff check . --fix

# Format
black .

# Type check
mypy analysis experiments qpu_types

# Run tests (add tests as needed)
pytest -q
```

Branching rules (for Codex and contributors):

- Analysis code: `codex/analysis`
- Experiments code: `codex/experiments`
- Hotfixes: `hotfix/<topic>`

## Repository Structure

```
analysis/           # Analysis workflows (fit/plot/extract parameters)
experiments/        # Experiment workflows (create/compile/run)
qpu_types/          # Qubit/bus models and operations (Transmon, Bus)
helper_functions/   # Utility functions (qubit params I/O, power helpers)
helper.py           # Backward-compat shim re-exporting helper_functions
operations.py       # Backward-compat shim re-exporting Transmon operations
examples/           # Example notebooks (imports updated to package layout)
pyproject.toml      # Build configuration and dependencies
```

Backwards compatibility:

- `from helper import ...` continues to work via `helper.py` shim.
- `from operations import TransmonOperations` continues to work via shim.
- `from qpu_types.transmon import TransmonQubit` remains valid (compat module),
  while the canonical path is `from qpu_types.Transmon.transmon import TransmonQubit` or
  `from qpu_types import TransmonQubit`.

## Usage Examples

Import an experiment and run it through the workflow:

```python
from laboneq.workflow.tasks import compile_experiment, run_experiment
from experiments.qubit_spectroscopy import experiment_workflow

compiled = compile_experiment(session, experiment_workflow(...))
result = run_experiment(session, compiled)
```

Run an analysis workflow:

```python
from analysis.resonator_spectroscopy import analysis_workflow

analysis_result = analysis_workflow(
    result=result,
    qubit=q0,
    frequencies=freqs_hz,
    options=analysis_workflow.options(),
).run()
params = analysis_result.output
```

Load and save qubit parameters:

```python
from helper import load_qubit_parameters, save_qubit_parameters

qubits = load_qubit_parameters(filename="latest", save_folder="./qubit_parameters")
save_qubit_parameters(qubits, save_folder="./qubit_parameters")
```

## Troubleshooting

- Import errors in notebooks: ensure you installed with `pip install -e .[dev]` and
  the active kernel uses the same environment.
- Case-sensitive imports: paths under `qpu_types/Transmon` are case-sensitive on Unix.
  Prefer `from qpu_types import TransmonQubit` where possible.
- LabOne Q versions: align `laboneq` and `laboneq-applications` versions in `pyproject.toml`
  with your environment. If you use a managed environment from ZI, keep to their pins.

## Contributing

- Follow branch rules above and open PRs with concise descriptions and checklists.
- Keep changes focused and avoid unrelated refactors.
- Run `ruff`, `black`, and `mypy` locally before pushing.
