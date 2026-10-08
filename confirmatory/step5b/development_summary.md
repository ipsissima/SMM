# Step 5B final development aggregate

**Status:** descriptive development audit only - not confirmatory inference.

- Subjects: 34/34
- Mean DeltaELPD(M3-M2): -0.011242902
- Median DeltaELPD(M3-M2): -0.006653943
- SD: 0.015374095
- IQR: [-0.013787789, -0.003209365]
- Range: [-0.078638528, 0.000946219]
- M3 > M2: 2/34 (0.059)
- Selected fit results valid: True (136 model/direction results)
- N1 nesting audit pass: True
- Max embedded-score error: 2.274e-13
- Minimum M2-M3 training gap: 0.000626019489
- Max transfer-identity error: 1.023e-18
- Frozen profile: 512 Sobol candidates / 16 polish starts / Haswell kernel

No bootstrap confidence interval, sign-flip p-value, band selection, or confirmatory decision is computed on development data.

The holdout boundary is enforced in code: any sub-044 or later JSON aborts aggregation.
