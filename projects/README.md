# Local Development Workspaces

Use `projects/` for local-only scratch work while developing reusable modules in
`qubit_experiment/`.

Typical use:

- create a workspace such as `projects/selectiveRIP-dev/`
- keep notebooks, one-off scripts, temporary configs, and private real-data access there
- import package code from `qubit_experiment.*`
- graduate anything reusable into package code and `tests/`

Suggested layout:

```text
projects/<workspace>/
  notebooks/
  scripts/
  data/        # local data or symlink; do not publish
  outputs/
  artifacts/
```

Rules:

- nothing under `projects/*` is meant to be committed
- do not store canonical library code here
- use this area to validate modules against real data before extracting the reusable
  parts into the public package
