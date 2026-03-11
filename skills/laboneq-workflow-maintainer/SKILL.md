---
name: laboneq-workflow-maintainer
description: Maintain and debug existing LabOne Q experiment/analysis workflows in this repository. Use for bug fixes, review follow-ups, and minor behavior changes under qubit_experiment/, experiments/, analysis/, and qpu_types/.
---

# LabOne Q Workflow Maintainer

## Scope

Use for maintenance work on existing modules:
- bug fixes from PR/review feedback
- workflow `Reference` and `OptionBuilder` failures
- acquisition/update contract drift
- small behavior extensions in existing experiment/analysis modules

Do not use for brand-new module creation or major redesigns. For those, use
`skills/module-builder/SKILL.md`.

## Triage Gate (before code edits)

1. Identify the failing path and exact file/function.
2. Classify root cause:
- workflow `Reference` misuse
- option builder typing/assignment issue
- acquisition/averaging contract mismatch
- `update_qpu`/parameter update path mismatch
- task-boundary violation (`workflow.log/comment/save_artifact`)
3. List paired modules that must stay in sync (`experiments/*` and `analysis/*`).

## Hard Rules

- Prefer canonical package edits under `qubit_experiment/*`; keep top-level
  compatibility modules import-compatible.
- Keep experiment and analysis behavior aligned when changing interfaces or outputs.
- In `@workflow.workflow` bodies, use workflow control flow (`workflow.if_`,
  `workflow.for_`), not Python `if/for/while`.
- Treat workflow outputs as `Reference`; avoid Python truthiness and container
  methods on unresolved values.
- Treat optional workflow inputs and workflow-parameter objects as potentially
  unresolved during graph construction; do not run `int(...)`, `bool(...)`,
  `isinstance(...)`, `x is None`, or attribute-based branching on them in the
  workflow body. Resolve them in helper tasks first.
- Set options with call style only (`opt.field(value)`), never assignment
  (`opt.field = value`).
- Do not assemble notebook-facing nested dict/list payloads inline in workflow
  returns when nested fields come from tasks or sub-workflows. Keep returns
  top-level, or assemble nested bundles in helper tasks or plain Python wrappers.
- When public maintenance code accepts both plain option objects and
  `OptionBuilder`s, normalize them explicitly before reading fields.
- Call `workflow.comment(...)`, `workflow.log(...)`, and
  `workflow.save_artifact(...)` only inside `@workflow.task`.
- Enforce acquisition/averaging constraints in `create_experiment(...)` with
  explicit `ValueError`.
- Persist updates only via
  `analysis_results.output["new_parameter_values"]` + `update_qpu(...)`.
- Do not use deprecated `update_qubits` in new maintenance code.
- Every bug fix should include a regression test under `tests/` unless explicitly
  blocked.

## Maintenance Workflow

1. Reproduce failure (or trace it from logs/review notes) and isolate the smallest
   failing path.
2. Implement a minimal fix in canonical modules; touch legacy wrappers only when
   required for compatibility.
3. Update paired experiment/analysis modules if interfaces or outputs changed.
4. Add or adjust regression tests targeting the failure mode.
   For notebook-facing payloads, include a recursive assertion that no nested value is
   a `laboneq.workflow.reference.Reference`.
5. Run fast checks:

```bash
ruff check .
pytest -q
```

6. Report root cause, changed contract (if any), and validation evidence.

## Quick PR Review Checklist

- Any workflow `Reference` consumed via Python control flow?
- Any workflow input or options object being cast/inspected directly in the workflow
  body?
- Any branch-local value read after a conditional chain?
- Any `OptionBuilder` field written by assignment?
- Any nested workflow return payload likely to leak `Reference` values into notebook
  output?
- Any analysis-derived output used when `do_analysis=False`?
- Any persistent update path missing `new_parameter_values` keys?
- Did the fix include a test that fails before and passes after?

## References

- `../module-builder/SKILL.md` (shared LabOne Q workflow contracts)
