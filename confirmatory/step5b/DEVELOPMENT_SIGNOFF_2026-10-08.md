# Step 5B final development sign-off

**PASS - HOLDOUT MAY OPEN**

This PASS certifies only that the frozen development/numerical prerequisites for opening the confirmatory holdout are satisfied. It is not an SMM empirical-success verdict.

## Provenance

- Canonical development run ID: `37621942072`
- Canonical development commit SHA: `901739e232bedd29785dbae5a416cb8fd346f388`
- Aggregate artifact: `SMM_STEP5B_DEVELOPMENT_AGGREGATE`
- Aggregate artifact ID: `11517163525`
- Aggregate artifact digest: `sha256:e4ea2329c7f460363f051ad1fd8ef6166de1d83879bb0dd61f8c8d9a4b593ffa`
- Final numerical profile: N1
- Robustness run ID: `37599448753`
- Numerical-method commit recorded by the frozen profile: `8ec566fd8f6e82d838a4c041db2a81d81ece4227`

## Completeness and information barrier

- 34/34 expected frozen-QC-passed development subjects are present.
- No subject sub-044 or later appears in the development aggregate.
- `confirmatory_inference_performed = false`.
- `holdout_subjects_seen = false`.
- The subject table exactly matches the frozen development PASS set.

## Numerical integrity

- Frozen profile: 512 Sobol candidates / 16 polish starts / Haswell production kernel.
- 136/136 selected M2/M3 x two-direction fit results satisfy the N1 success schema.
- Every development training block satisfies the exact M3-in-M2 nesting checks enforced by the aggregate.
- Max embedded-score error: 2.274e-13.
- Minimum M2-M3 training gap: 0.000626019489346.
- Max transfer-identity error: 1.023e-18.
- The aggregate embeds exactly the frozen N1 numerical profile.

## Descriptive development result

- Mean DeltaELPD(M3-M2): -0.0112429018231
- Median DeltaELPD(M3-M2): -0.0066539431952
- SD: 0.0153740953753
- IQR: [-0.0137877888132, -0.00320936494452]
- Range: [-0.0786385281392, 0.000946219262232]
- M3 > M2: 2/34 (0.058824)

These development quantities are descriptive only and did not alter the model, comparator, endpoint, QC, frequency range, subject split, or confirmatory success criterion.

## Permanent repository file hashes

- `development_summary.json`: `sha256:e346e4e0c92ecd2a005a9c2a4d57d77489b8cbf1d625aa957897a93a6e2b051d`
- `development_subject_results.csv`: `sha256:51f7edc5e195141b6fef393ec83f89f344499f24a9fa466ac031a965786cd936`
- `development_optimizer_diagnostics.csv`: `sha256:5b742cf700576de55fb1c211da1598d5583fb9e1584e325e121a4d7337ad1036`
- `development_summary.md`: `sha256:7748106cf4c96677f1c2bfcd019d33d8a1b19e74659859961bb81ff12fc6b720`

The aggregate artifact digest above remains the canonical archive-level digest of the workflow output.

**Sign-off verdict: PASS - HOLDOUT MAY OPEN.**
