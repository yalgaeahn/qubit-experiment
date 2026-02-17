# 2Q Quantum State Tomography (Superconducting Qubits)
## Readout Error Mitigation + MLE Workflow (Practical Guide)

This document describes the complete experimental + analysis pipeline for
2‑qubit quantum state tomography using:

- 9 Pauli measurement settings
- assignment (readout) error mitigation
- Maximum Likelihood Estimation (MLE)
- raw counts only (no expectation preprocessing)

This version intentionally avoids LaTeX so it renders cleanly in VSCode Markdown.

------------------------------------------------------------
INDEX NOTATION
------------------------------------------------------------

s : measurement setting (XX, XY, ..., ZZ)      -> 9 total
k : ideal computational basis outcome          -> {00,01,10,11}
i : measured outcome after readout noise
n[s,i] : raw counts (ONLY real experimental data)

------------------------------------------------------------
STEP 1 — Measurement settings
------------------------------------------------------------

Run exactly these 9 settings:

XX  XY  XZ
YX  YY  YZ
ZX  ZY  ZZ

For each setting s:
1. Apply pre‑rotation U_s
2. Measure in Z basis
3. Collect raw counts:
       n[s,00], n[s,01], n[s,10], n[s,11]

IMPORTANT:
- Do NOT convert to probabilities
- Do NOT compute expectation values

------------------------------------------------------------
STEP 2 — Assignment matrix A (readout calibration)
------------------------------------------------------------

This is done ONCE. Not per setting.

For each ideal basis state k:

prepare |k>
measure many shots
record counts c[i]^(k)

Compute:

N[k] = sum_i c[i]^(k)

A[i,k] = c[i]^(k) / N[k]

Meaning:
A[i,k] = P(measured i | ideal k)

Each column of A must sum to 1.

NOTE:
A is independent of s because readout noise is hardware‑level,
not basis‑dependent.

------------------------------------------------------------
STEP 3 — Ideal POVM construction
------------------------------------------------------------

For each setting s:

E_ideal[s,k] = U_s† |k><k| U_s

This defines the measurement basis.

------------------------------------------------------------
STEP 4 — Include readout error in POVM
------------------------------------------------------------

E_meas[s,i] = sum_k A[i,k] * E_ideal[s,k]

Now the POVM includes assignment noise.

------------------------------------------------------------
STEP 5 — Predicted probabilities
------------------------------------------------------------

p[s,i](rho) = Tr(E_meas[s,i] * rho)

These are computed by the model, NOT measured.

------------------------------------------------------------
STEP 6 — Log‑likelihood for MLE
------------------------------------------------------------

logL(rho) = sum_s sum_i n[s,i] * log( p[s,i](rho) )

Key fact:
Only n[s,i] comes from experiment.
Everything else is computed.

------------------------------------------------------------
STEP 7 — Parameterize rho (physical state)
------------------------------------------------------------

Use Cholesky parameterization:

rho = T†T / Tr(T†T)

Optimize parameters of T to maximize logL.

Use optimizers such as:
- L‑BFGS
- Adam
- Nelder‑Mead

------------------------------------------------------------
STEP 8 — Final outputs
------------------------------------------------------------

rho_hat = reconstructed density matrix

Then compute observables:

<XI> = Tr((X ⊗ I) rho_hat)
<XX> = Tr((X ⊗ X) rho_hat)
fidelity, purity, etc.

------------------------------------------------------------
IMPORTANT RULES
------------------------------------------------------------

✓ use 9 settings only
✓ use raw counts only
✓ calibrate A once
✓ include A inside POVM
✓ use MLE

✗ do not average expectation values
✗ do not invert A
✗ do not use linear inversion alone

------------------------------------------------------------
END
------------------------------------------------------------
