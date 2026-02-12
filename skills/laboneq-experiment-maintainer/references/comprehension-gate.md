# Comprehension Gate for New Experiment/Analysis Requests

Run this before implementing new modules or major redesigns.

## Required pre-code output (must be confirmed by user)
1. Objective
- What is measured/estimated/optimized and why.

2. Physics and measurement model
- Core assumptions and the mapping from measured data to target metrics.

3. Interface contract
- Inputs: quantum elements, sweep variables, acquisition type/mode, required options.
- Outputs: metrics, plots, and update payload keys.

4. Update policy
- Which values are runtime-only (temporary) vs persisted (`new_parameter_values`).

5. Validation plan
- Synthetic sanity check and real-data smoke check.
- Pass/fail criteria (normalization, finite values, PSD/trace checks, etc.).

6. Exclusions
- Explicitly list what is out-of-scope for this change.

## Traceability table (required in implementation report)
- Requirement -> file/function -> output field -> validation evidence.

## Stop condition
- Do not start code edits until the user confirms the pre-code output.
