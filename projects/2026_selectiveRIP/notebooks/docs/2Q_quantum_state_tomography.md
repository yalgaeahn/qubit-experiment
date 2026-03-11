# 2-Qubit Quantum State Tomography with Readout Error Mitigation + MLE

## Overview

This document describes a complete experimental and data-analysis workflow for 2-qubit quantum state tomography (QST) with readout assignment error mitigation and maximum likelihood estimation (MLE).  
The procedure is tailored for superconducting qubits or any qubit platform with readout assignment noise, using informationally complete Pauli measurement settings and integrating readout errors directly into the MLE likelihood model.

Key ideas:
- Calibration of readout noise via assignment/confusion matrix \(A_{ik}\).
- 9 Pauli measurement settings for 2-qubit tomography are informationally complete.
- Direct inclusion of assignment noise into the POVM for MLE fitting.
- Raw counts (not preprocessed expectation values) are used in the likelihood.

---

## Notation

- \(s\): measurement setting label, one of the 9 Pauli combinations:
- \(k\): ideal outcome index in computational basis \(\{00,01,10,11\}\)
- \(i\): measured outcome index after assignment noise
- \(n_{s,i}\): raw counts for setting \(s\), outcome \(i\)
- \(p_{s,i}(\rho)\): predicted probability under density matrix \(\rho\)
- \(E^{ideal}_{s,k}\): ideal POVM element for setting \(s\), outcome \(k\)
- \(E^{meas}_{s,i}\): noisy POVM element for setting \(s\), measured outcome \(i\)
- \(A_{ik}\): assignment/confusion matrix mapping ideal outcome \(k\) to measured outcome \(i\)

---

## STEP 1 — Measurement Settings

Perform the following 9 Pauli measurements on identically prepared states:

Each setting:
1. Apply appropriate pre-rotation \(U_s\) before readout to measure in the desired basis.
2. Collect raw counts \(n_{s,i}\) for all four measurement outcomes (00,01,10,11).
3. Repeat enough shots (e.g., 5k–20k) for statistical significance.

These 9 settings form an **informationally complete** set for 2-qubit tomography.

---

## STEP 2 — Calibration of Assignment Matrix \(A_{ik}\)

**Only one calibration procedure is needed (no dependence on \(s\))**:

1. Prepare the computational basis states \(|k\rangle\) for \(k\in\{00,01,10,11\}\).
2. For each prepared \(|k\rangle\), perform readout many times and gather raw counts \(c_i^{(k)}\) for outcomes \(i\).
3. Compute total shots:
$$
   N^{(k)} = \sum_i c_i^{(k)}
$$
4. Compute conditional probabilities (assignment matrix):  

   $$ A_{ik}
   =
   \frac{c_i^{(k)}}{N^{(k)}}$$

   - column index \(k\) = prepared ideal outcome
   - row index \(i\) = measured outcome

   Each column of \(A\) sums to 1.

This step characterizes readout noise such as misclassification and crosstalk.

---

## STEP 3 — Construct Ideal POVM for Setting \(s\)

For each measurement setting \(s\), define the ideal POVM elements in terms of basis \(k\):

1. Define computational basis projectors:
   $$
   \Pi_k = |k\rangle\langle k|
   $$
2. Apply pre-rotation \(U_s\) for setting \(s\):
   $$
   E^{\text{ideal}}_{s,k}
   =
   U_s^\dagger \Pi_k U_s
   $$

These are the **noise-free** POVM elements for setting \(s\).

---

## STEP 4 — Incorporate Readout Error into POVM

Use assignment matrix \(A_{ik}\) to build **noisy measurement operators**:

$$
E^{\text{meas}}_{s,i}
=
\sum_{k}
A_{ik}
E^{\text{ideal}}_{s,k}
$$

This defines the effective POVM that includes readout error.  
It is used to compute the predicted probabilities in the likelihood model.

---

## STEP 5 — Probability Model for MLE

For given state \(\rho\), predicted probability for setting \(s\) and outcome \(i\) is:

$$
p_{s,i}(\rho)
=
\mathrm{Tr}\left(E^{\text{meas}}_{s,i}\rho\right)
$$

These probabilities must satisfy
\(\sum_i p_{s,i}(\rho)=1\) for each setting \(s\).

---

## STEP 6 — Log-Likelihood Formulation

Using raw counts \(n_{s,i}\), the log-likelihood function is:

$$
\log L(\rho)
=
\sum_{s}
\sum_{i}
n_{s,i}\,
\log p_{s,i}(\rho)
$$

This is the function to maximize over valid density matrices.

MLE finds the density matrix \(\hat\rho\) that **best explains the observed counts** under the measurement model.

---

## STEP 7 — Parameterization of \(\rho\)

To enforce physicality (Hermitian, PSD, unit trace), parameterize \(\rho\) using Cholesky or similar:

$$
\rho(\theta)
=
\frac{T(\theta)^\dagger T(\theta)}{
\mathrm{Tr}(T(\theta)^\dagger T(\theta))
}
$$

Where \(T(\theta)\) is lower-triangular with appropriate free parameters \(\theta\).  
Minimize negative log-likelihood:

$$
\min_\theta
-
\sum_{s,i}
n_{s,i}
\log
\mathrm{Tr}\left(E^{\text{meas}}_{s,i}\rho(\theta)\right)
$$

Using any gradient-based optimizer or convex solver.

---

## STEP 8 — Validation and Postprocessing

1. Verify \(\hat\rho\) is PSD and trace-1.
2. Compute expectation values:
   $$
   \langle O\rangle = \mathrm{Tr}(O \hat\rho)
   $$
   for observables \(O\) of interest (e.g., Pauli correlators).
3. Compute fidelity to a target state if known.
4. Bootstrap or Monte Carlo for error bars if needed.

---

## Notes & Best Practices
 
- Assignment matrix calibration is typically done once per readout configuration. Recalibrate if readout drifts.
- The model can be extended to more qubits with \(3^N\) Pauli settings for full tomography.
- MLE is preferred over linear inversion for physicality and statistical optimality.
- Overcomplete measurements (using more than minimum settings) improve noise robustness.

---

## References

- Quantum state tomography reconstructs the density matrix from repeated measurements in multiple bases. (Wikipedia: Quantum tomography)
- Readout error mitigation methodology integrates measurement calibration into state reconstruction. (Nature Communications)
- Frameworks such as Qiskit Experiments include readout error mitigated tomography tools requiring characterization experiments and Pauli measurement circuits. (Qiskit Experiments docs)
