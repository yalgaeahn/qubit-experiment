---
name: module-builder
description: Create new LabOne Q experiment/analysis modules in this repository by grounding design against the installed `.venv` `laboneq_applications` contracts before coding. Use when the user asks to add a new workflow/module from scratch, add a new tune-up experiment type, or perform a major interface redesign under `experiments/` and `analysis/`.
---

# Module Builder (LabOne Q)

## Scope

Use only for new module creation or large redesigns.

If the request is only a bug fix or minor edit in existing modules, do not use this
skill; handle it as a normal repository edit task.

## Mandatory Venv Grounding Gate

Run this before writing any code:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
.venv/bin/python skills/module-builder/scripts/snapshot_laboneq_applications.py \
  --output skills/module-builder/references/laboneq-applications-venv-snapshot.md
```

Do not start implementation unless this succeeds.

Before coding, provide:
- package version + package path from snapshot
- selected template module(s) from installed package (experiment is required; analysis
  may be dedicated, shared, or intentionally omitted)
- requested interface to template-contract mapping
- update policy (`temporary_parameters` vs persistent updates, if update is supported)
- acquisition/averaging constraints and explicit error policy

Use `references/new-module-precode-template.md` for the exact format.

## Build Workflow

1. Run the Venv Grounding Gate command and read the generated snapshot.
2. Select nearest template(s) from installed package:
- `laboneq_applications.experiments.*`
- `laboneq_applications.analysis.*`
3. Decide analysis strategy for the new module:
- dedicated new analysis module
- shared existing analysis module
- no analysis path in workflow
4. Produce and confirm the pre-code brief with user.
5. Implement module(s) according to the chosen strategy.
6. Add/update tests and notebook validation cells if requested.
7. Run syntax/tests and report evidence.

## Contract Rules

- Add a module-level docstring at the top of every new module (`experiments/*.py`,
  `analysis/*.py`) with a short purpose/sequence summary and key contracts
  (`acquisition_type` constraints, update behavior, analysis linkage if applicable).
- Keep workflow entrypoint on `@workflow.workflow`.
- Keep experiment builder on `@workflow.task` + `@dsl.qubit_experiment`.
- Apply runtime overrides via `temporary_qpu` and
  `temporary_quantum_elements_from_qpu`.
- For update-capable workflows only, apply persistent updates through
  `analysis_results.output["new_parameter_values"]` + `update_qpu(...)`.
- Keep tune-up defaults explicit: `do_analysis=True`, `update=False`.
- Do not use deprecated `update_qubits` in new code.
- In `@workflow.workflow` body, treat task/workflow outputs as `Reference` and keep a
  workflow-safe subset only: task/sub-workflow calls, `with workflow.if_`,
  `with workflow.elif_`, `with workflow.else_`, `with workflow.for_`,
  `workflow.return_`.
- Do not use Python control flow in workflow body (`if`, `for`, `while`) or Python
  truthiness checks on unresolved references (`bool(...)`, `len(...)`, `in`,
  `isinstance(...)`, `x is None`).
- Do not call container methods on unresolved references
  (`.get()`, `.items()`, `.keys()`, `.values()`, `.append()`, `.extend()`, `.pop()`).
- Build compound conditions as nested `workflow.if_` blocks. If real Python logic is
  needed, move it to a helper `@workflow.task(save=False)` and branch on that task
  output.
- For list/dict accumulation inside workflow-level loops, use helper tasks
  (`@workflow.task(save=False)`), not direct Python mutation in workflow body.
- Initialize branch-dependent outputs before `workflow.if_` blocks and only return/use
  values guaranteed to be resolved on every execution path.
- Do not read or forward values created inside `workflow.if_/elif_/else_` outside
  that conditional chain unless assigned on every path.
- When routing analysis by branch, perform `analysis_workflow(...)` and
  `update_qpu(...)` inside each branch.
- If a shared post-branch value is required, compute it via a helper task executed on
  all paths with a stable schema.
- Avoid cross-branch aliasing (for example assigning `analysis_results` in separate
  branches, then reading `analysis_results.output` after the chain).
- Use `==` for comparisons with references; do not use identity checks (`is`).
- Set options only with `OptionBuilder` call style (`opt.field(value[, selector])`);
  never assign with `opt.field = value`.
- Call `workflow.comment(...)`, `workflow.log(...)`, and
  `workflow.save_artifact(...)` only inside `@workflow.task`.
- Enforce acquisition/averaging contract in `create_experiment(...)` with explicit
  `ValueError`.

### Branch-Safe Pattern

- Forbidden pattern:

```python
with workflow.if_(cond):
    analysis_results = analysis_a(...)
with workflow.else_():
    analysis_results = analysis_b(...)
update_qpu(qpu, analysis_results.output["new_parameter_values"])
```

- Required pattern:

```python
with workflow.if_(cond):
    analysis_results = analysis_a(...)
    with workflow.if_(options.update):
        update_qpu(qpu, analysis_results.output["new_parameter_values"])
with workflow.else_():
    analysis_results = analysis_b(...)
    with workflow.if_(options.update):
        update_qpu(qpu, analysis_results.output["new_parameter_values"])
```

## Validation Checklist

- Experiment module interface and analysis strategy (dedicated/shared/none) are
  internally consistent.
- New module files include a top-level docstring that matches implemented behavior.
- If update is supported, analysis/update path returns and applies expected keys via
  `old_parameter_values` / `new_parameter_values` and `update_qpu`.
- Workflow-body audit confirms no Python `if/for/while` and no forbidden
  `Reference` operations listed above.
- Options setup uses `OptionBuilder` call style only and contains no assignment-style
  option writes.
- Smoke run or tests execute without `Reference`/`OptionBuilder` type errors.
- Report includes template-module mapping and validation evidence.
- No variable first created in a branch is consumed outside the conditional chain
  unless guaranteed on all paths.
- Each conditional route is smoke-tested at least once (for example `echo=True` and
  `echo=False`) to catch unresolved references.
- When `do_analysis=False`, no analysis-derived output is accessed.

## Reference Error Triage

- Error: `Iterating a workflow Reference is not supported.`
  Action: replace Python iteration over workflow values with `workflow.for_(...)` and
  move list/dict mutation to helper tasks.
- Error: `Result for '...' is not resolved.`
  Action: ensure every branched value is initialized before conditionals and assigned
  on all paths; avoid Python control flow in workflow body.
  If the value is branch-specific, move its consumption (including `update_qpu`) into
  the same branch rather than sharing a post-branch variable.
- Error: `Setting options by assignment is not allowed.`
  Action: replace assignment (`opt.x = ...`) with call style (`opt.x(...)`).

## Quick Branch-Safety Review

- Any branch-local variable used after branch end?
- Any `analysis_results.output[...]` consumed outside its producing branch?
- Any analysis output accessed when analysis may be disabled?
- Any task input sourced from a branch that may not execute?

## References

- `references/new-module-precode-template.md`
- `references/laboneq-applications-venv-snapshot.md` (generated at runtime)

Script:
- `scripts/snapshot_laboneq_applications.py`
