# LabOne Q User Manual Guards

Use this guide when coding or debugging workflows in this repository.

## Primary docs
- User Manual index: https://docs.zhinst.com/labone_q_user_manual/index.html
- Workflow syntax and `Reference`: https://docs.zhinst.com/labone_q_user_manual/core/functionality_and_concepts/07_workflow/tutorials/02_workflow_syntax.html
- Workflow reference APIs: https://docs.zhinst.com/labone_q_user_manual/core/reference/workflow.html
- Workflow results (`TaskView`): https://docs.zhinst.com/labone_q_user_manual/core/reference/workflow/result.html
- DSL enums (`AcquisitionType`, `AveragingMode`): https://docs.zhinst.com/labone_q_user_manual/core/reference/dsl/enums.html
- Averaging and sweeping concepts: https://docs.zhinst.com/labone_q_user_manual/core/functionality_and_concepts/03_sections_pulses/concepts/04_averaging_sweeping.html
- Applications Library options defaults: https://docs.zhinst.com/labone_q_user_manual/applications_library/reference/experiments/options.html
- Parameter updating task docs: https://docs.zhinst.com/labone_q_user_manual/applications_library/reference/tasks/parameter_updating.html
- Logbook/recording tutorial: https://docs.zhinst.com/labone_q_user_manual/applications_library/tutorials/sources/logbooks.html
- Release notes: https://docs.zhinst.com/labone_q_user_manual/release_notes/index.html

## Rules to apply
1. Workflow-body object model
- Values from workflow/task calls in workflow body are `Reference` objects.
- Do not run dict-like operations (`.items()`, `.values()`, `.get()`) on unresolved references.

2. Result extraction model
- `workflow_result.tasks` is a `TaskView`/sequence interface, not always a plain dict.
- Use key/index access patterns from docs (`tasks["name"]`, `tasks["name", :]`, `tasks[0]`).

3. Option defaults and override intent
- Tune-up style workflows default to `do_analysis=True`, `update=False`.
- Explicitly set overrides in notebooks/scripts to avoid silent behavior drift.

4. Acquisition contract enforcement
- Validate acquisition settings at experiment creation time (type + averaging contract).
- Be explicit when analysis expects complex IQ integration data.

5. Update path discipline
- Persist only through `new_parameter_values` and apply with `update_qpu`.
- Keep `old_parameter_values`/`new_parameter_values` naming stable for update-capable workflows.
- Avoid introducing new `update_qubits` usage.

6. Temporary vs persistent parameters
- Use temporary copies for runtime-only overrides; keep persistent updates separate.
- Never mix temporary overrides into persistent update payload by accident.

7. Debug sequence
- Use `run(until="...")` for isolation.
- Inspect `workflow_result.tasks[...]` directly.
- Keep artifact saving inside tasks; notebook plotting should not call workflow artifact APIs.

8. Version drift control
- When touching workflow/update APIs, check release notes for deprecations/behavior changes.
