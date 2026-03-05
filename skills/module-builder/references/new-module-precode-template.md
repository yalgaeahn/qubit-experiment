# New Module Pre-code Brief Template

Use this exact template before implementing a new experiment/analysis module.

## 1) Snapshot metadata
- package_version:
- package_path:
- generated_utc:

## 2) Requested objective
- what to measure/estimate:
- why this module is needed:

## 3) Template mapping from installed package
- experiment template module:
- analysis strategy (dedicated new module / shared existing module / none):
- analysis template module (if applicable):
- why these are the nearest contracts:

## 4) Interface contract
- inputs (qubits, sweep args, options):
- outputs (metrics, payload keys, artifacts):
- workflow option defaults:
- planned top-of-file module docstring (3-6 lines):

## 5) Update policy
- runtime-only overrides (`temporary_parameters`):
- persistent updates (if update is supported; `new_parameter_values` + `update_qpu`):

## 6) Acquisition and averaging policy
- allowed acquisition type(s):
- allowed averaging mode(s):
- explicit invalid combinations and raised error:

## 7) Reference-safety plan
- workflow body control-flow plan (`workflow.if_` / `workflow.for_` only):
- helper tasks needed for Python logic or list/dict mutation:
- branch output initialization plan (avoid unresolved references):
- update path key access plan (`analysis_results.output["new_parameter_values"]`):

## 8) Validation plan
- smoke run:
- error-contract test:
- success criteria:

## 9) Out of scope
- excluded items for this iteration:
