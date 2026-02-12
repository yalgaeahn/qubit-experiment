---
name: laboneq-experiment-maintainer
description: Build, modify, and debug LabOne Q experiment/analysis workflows in this repository, especially for fixed-transmon calibration/tomography/readout tasks. Use when working on files under experiments/, analysis/, qpu_types/, or selectiveRIP notebooks, and when fixing workflow OptionBuilder/Reference issues, acquisition-mode mismatches, parameter update paths, and validation plotting.
---

# LabOne Q Experiment Maintainer

## Scope
Use for development and debugging of:
- `experiments/*.py`
- `analysis/*.py`
- `qpu_types/fixed_transmon/*.py`
- validation notebooks under `examples/selectiveRIP/*.ipynb`

## Hard Rules
- Modify paired experiment/analysis modules together when behavior changes.
- For new module requests, run a Comprehension Gate before coding (no implementation until user confirms the brief).
- Enforce acquisition contract in `create_experiment(...)` with explicit errors.
- Keep workflow branching that depends on values inside `@workflow.task` (avoid `Reference` logic in workflow body).
- Treat workflow-body values as `Reference`; do not call dict methods on unresolved references.
- Persist calibration values only through `new_parameter_values` and apply with `update_qpu(...)`.
- Save workflow artifacts only inside `@workflow.task`.
- In notebooks, append validation cells; do not rewrite existing cells unless requested.
- Keep workflow option values concrete and type-safe (`bool`, `str`, `int`, etc.); do not pass unresolved `Reference` into option setters.
- Keep tune-up option defaults explicit (`do_analysis=True`, `update=False`) unless intentionally overridden.

## Workflow
1. If this is a new experiment/analysis request, complete Comprehension Gate and get user confirmation.
2. Read both experiment and analysis files for the target flow.
3. Implement behavior change in experiment sequence/options and analysis outputs consistently.
4. Verify update path (`old_parameter_values` / `new_parameter_values`) for persisted parameters.
5. Verify `temporary_parameters` handling and `qpu.copy_qubits()` usage when runtime overrides are needed.
6. Add or update notebook validation cells (pass/fail style).
7. Run quick syntax checks before finishing.

## Validation
- Workflow executes without `OptionBuilder`/`Reference` type errors.
- Matrix/count outputs are finite and normalized on the correct axis by definition.
- Reconstructed density matrix checks: Hermitian, trace≈1, min eigenvalue ≥ tolerance.
- If `update=True`, expected parameter keys are present in `new_parameter_values`.
- If workflow output is needed in notebooks, materialize task outputs and handle `workflow_result.tasks` as `TaskView`/sequence (not plain dict).

## Do Not
- Do not call `workflow.save_artifact` from notebook helper functions.
- Do not silently coerce unsupported acquisition modes.
- Do not change existing notebook cells when append-only validation is sufficient.
- Do not use deprecated `update_qubits` in new code; use `update_qpu`.

## References
For concrete error signatures and fixes, see `references/common-errors.md`.
For doc-backed workflow contracts and update semantics, see `references/applications-library-contracts.md`.
For User Manual guardrails (Reference/TaskView/acquisition/debug contracts), see `references/laboneq-user-manual-guards.md`.
For request understanding and implementation standardization, see `references/comprehension-gate.md`.

Reference trigger policy:
- Core layer first (`laboneq.dsl`, `Experiment`, `acquire_loop_rt`, `AcquisitionType`, compile/session/device mapping): read `references/laboneq-user-manual-guards.md`.
- Applications layer first (`laboneq_applications`, `experiment_workflow`, `.options()`, `do_analysis`, `update`, `update_qpu`, `temporary_parameters`): read `references/applications-library-contracts.md`.
- New module request (`make new experiment/analysis`, `add workflow from scratch`, major redesign): read `references/comprehension-gate.md` first, then continue with layer-specific references.
- Boundary changes (experiment + analysis + update path edited together): read both references, then cross-check with `references/common-errors.md`.
- Error-first fallback: `Reference`/`OptionBuilder`/`TaskView`/`update_qpu` issues -> applications contracts + common errors; compile/device/signal-map issues -> user-manual guards + common errors.
