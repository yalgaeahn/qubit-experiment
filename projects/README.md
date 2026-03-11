# Local Development Workspaces

Use `projects/` for project-specific workspaces while developing reusable modules
in `qubit_experiment/`.

Typical use:

- keep shared project assets in a workspace such as `projects/2026_selectiveRIP/`
- create extra scratch workspaces such as `projects/selectiveRIP-dev/` when needed
- keep notebooks, one-off scripts, temporary configs, and real-data access there
- import package code from `qubit_experiment.*`
- graduate anything reusable into package code and `tests/`

Suggested layout:

```text
projects/<workspace>/
  notebooks/
  scripts/
  custom_qubit_experiment/
  qpu_parameters/
  data/        # local data or symlink; do not publish
  outputs/
  artifacts/
```

Notebook example:

```python
from qubit_experiment.devpaths import workspace_path

descriptor_path = workspace_path(
    "configs/descriptors/1port.yaml",
    workspace="2026_selectiveRIP",
)
```

Rules:

- do not store canonical library code here
- use this area to validate modules against real data before extracting the reusable
  parts into `qubit_experiment/`
- shared project workspaces may be tracked; private raw data and transient outputs
  should stay ignored
