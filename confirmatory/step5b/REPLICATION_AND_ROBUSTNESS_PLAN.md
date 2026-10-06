# Step 5B replication and robustness plan

**Status:** pre-result planning only. The primary holdout result remains the sole primary test.

## Frozen replication order

After the primary session-1 / EyesClosed / acq-pre result is complete, apply the unchanged mechanism and numerical implementation in this order:

1. session-1 / EyesOpen / acq-pre;
2. session-1 / EyesClosed / acq-post;
3. session-1 / EyesOpen / acq-post;
4. session-2 / EyesClosed / acq-pre;
5. remaining session-2 conditions in metadata-defined order.

No replication result may be used to revise and rerun the primary holdout.

## Invariants across replications

The following remain unchanged unless a condition makes a purely mechanical filename/metadata substitution necessary:

- M2 and M3 equations;
- K-to-QIF mapping;
- astroglial coefficients/topology;
- neuronal backbone;
- DK68 structural prior and normalization;
- forward model;
- 1-40 Hz primary frequency support;
- empirical CSD definition;
- two chronological CV directions;
- seed 97;
- 32 Sobol candidates;
- 4 L-BFGS-B polish starts;
- deterministic single-thread environment;
- CSD regularization;
- subject-level DeltaELPD definition.

## QC

The same preprocessing and QC rules are applied condition-wise. Missing recordings are not failures and are reported as unavailable. Recordings that fail frozen QC are excluded for the frozen reason; thresholds are not relaxed to increase replication n.

## Reporting

For every replication condition report:

- metadata-available n;
- QC-pass n;
- mean/median/SD DeltaELPD(M3-M2);
- fraction of subjects with Delta > 0;
- uncertainty interval for the group mean using the same subject-level bootstrap machinery where applicable;
- direction relative to the primary result;
- optimizer diagnostics;
- missingness/QC table.

The strongest replication pattern is the same positive M3-vs-M2 direction across eyes-open/closed and longitudinal data. Failure to reproduce the primary sign must be reported plainly.

## Interpretation barrier

A primary PASS followed by weak/heterogeneous replications supports condition- or state-dependent predictive value, not universal SMM superiority.

A primary FAIL cannot be converted into a success claim by a favorable secondary condition. Favorable replications after primary failure are hypothesis-generating/secondary evidence unless independently preregistered otherwise.

## Robustness analyses after frozen replications

Robustness is subordinate to the primary and replication sequence.

Permitted planned checks include:

- descriptive secondary PSD, debiased squared wPLI and imaginary coherency;
- metastability and network-persistence statistics;
- descriptive aperiodic slope under the frozen definition;
- forward-model limitations/sensitivity already declared;
- alternative neuronal backbones only as explicitly labeled robustness models;
- sensitivity to allowed nuisance/network quantities within frozen bounds.

Not permitted:

- selecting a favorable EEG band after primary results;
- weakening M2;
- changing exclusion thresholds;
- changing the K-to-QIF mapping;
- introducing centimeter-scale astroglial edges;
- retuning glial physiology on the holdout;
- redefining the primary success rule.

## Permanent outputs

Each condition should produce the same four lightweight permanent products:

1. subject result CSV;
2. optimizer diagnostic CSV;
3. group summary JSON;
4. Markdown result summary.

The provenance manifest must record workflow run, artifact ID/digest and the commit that generated each condition.
