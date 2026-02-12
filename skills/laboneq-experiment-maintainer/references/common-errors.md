# Common Errors and Fixes (LabOne Q in this repo)

## 1) `TypeError: 'Reference' object is not callable`

Typical causes:
- Treating workflow `Reference` like a normal dict/object during workflow construction.
- Calling `.items()` / `.get()` / function-call patterns on unresolved outputs.

Fix:
- Move value resolution to `@workflow.task` and return concrete python objects.
- In notebooks, unwrap nested outputs and reconstruct from task outputs when needed.

## 2) Option builder errors during `.options()`

Typical symptom:
- Type validator failure when assigning workflow options from another option reference.

Fix:
- Do not push `Reference`-typed values into typed option setters in workflow body.
- Resolve bool/string/int in dedicated task and pass concrete results to analysis workflow.

## 3) `workflow.save_artifact` runtime error outside tasks

Symptom:
- `Workflow artifact saving is currently not supported outside of tasks.`

Fix:
- Keep artifact saving inside `@workflow.task` functions only.
- Notebook-side plotting should use matplotlib directly without `workflow.save_artifact`.

## 4) Acquisition contract mismatch

Symptom:
- Analysis assumes IQ integration but experiment runs DISCRIMINATION/SPECTROSCOPY, or vice versa.

Fix:
- Add explicit guards in `create_experiment(...)` for `AcquisitionType` and `AveragingMode`.
- Keep defaults and analysis model consistent.

## 5) Parameter not updated even with `update=True`

Symptom:
- Printed metric exists, but QPU parameter remains unchanged.

Fix:
- Ensure value is present under `new_parameter_values[uid][param_name]`.
- Ensure workflow uses `update_qpu(...)` on that dict.

## 6) Plot readability vs physics confusion for phase

Issue:
- Wrapped phase appears step-like near ±pi.

Guideline:
- Distinguish computation data from display transforms.
- Keep calculations on corrected/unwrapped data.
- Apply display-only alignment/wrapping in plotting code and label clearly.
