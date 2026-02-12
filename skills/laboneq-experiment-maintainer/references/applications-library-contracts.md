# LabOne Q Applications Library Contracts (Doc-backed)

Use this file when implementing or debugging workflow structure and update semantics.

## Official docs
- Applications Library index: https://docs.zhinst.com/labone_q_user_manual/applications_library/index.html
- Experiment workflow details: https://docs.zhinst.com/labone_q_user_manual/applications_library/how-to-guides/01_experiment_workflow.html
- Qubit parameter updates (`update_qpu` task): https://docs.zhinst.com/labone_q_user_manual/applications_library/reference/tasks/parameter_updating.html
- Task/workflow options: https://docs.zhinst.com/labone_q_user_manual/core/reference/workflow.html

## Contracts to enforce in this repo
1. Experiment workflow return contract
- Workflow output should expose parameter-update payloads when calibration updates are part of the flow.
- For update-capable workflows, keep `old_parameter_values` and `new_parameter_values` consistently named.

2. Update semantics
- Prefer `update_qpu(qpu=qpu, parameters=new_parameter_values)`.
- Treat `update_qubits(...)` as deprecated; do not add new usage.
- `update=True` only means "apply prepared numeric update payload"; no payload means no persistence.

3. Option semantics
- Use typed options and pass concrete values (`bool`, `str`, `int`, etc.).
- Do not assign unresolved workflow `Reference` objects to typed option fields.

4. Temporary parameter handling
- Runtime overrides should be applied by creating copied qubits from temporary parameters (for example through `qpu.copy_qubits()` patterns used in this repo).
- Keep persistent updates separate from temporary overrides.

5. Debugging patterns
- Inspect workflow task outputs via `workflow_result.tasks[...]`.
- For partial debugging, use `run(until="task_or_subworkflow_name")`.
- Keep artifact saving inside `@workflow.task`; notebook-side helpers should only use matplotlib/file IO directly.

6. Readout/state-preparation consistency
- Ensure prepare-state labels and acquisition mode assumptions are explicit and aligned across experiment and analysis.
- If analysis expects complex IQ integration data, enforce acquisition type/mode contract in experiment creation.
