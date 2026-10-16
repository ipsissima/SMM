# SMM Definitive Reconstruction — Step 5B
## Pre-EEG empirical-analysis lock

**Status:** SCIENTIFICALLY FROZEN BEFORE RAW EEG SIGNAL INSPECTION  
**Date:** 2026-10-05  
**Dataset:** OpenNeuro `ds005385`, version 1.0.3  
**Raw EEG samples used in this step:** **none**

---

## Executive decision

The mechanistic reconstruction is over.

There is no planned Step 6 of hidden biology or another round of equation
construction. From this point the project changes phase from **model building**
to **falsification**.

The remaining sequence is finite:

1. **Step 5B — this lock:** freeze split, forward model, preprocessing, features,
   likelihood, null hierarchy, and success criterion.
2. **Development:** run the historically exposed subjects `sub-001..043`; the
   metadata-clean primary development subset is `n=38` and five prespecified
   late-trigger subjects are sensitivity-only.
3. **One development freeze:** fix legitimate optimizer/numerical tolerances once.
   The mechanism and primary endpoint cannot be redesigned to chase fit.
4. **Confirmatory holdout:** `sub-044..608`, primary condition session-1,
   eyes-closed, pre-cognitive-block.
5. **Robustness/comparators + manuscript.**

If the data falsify SMM, the project can reopen mechanistically **because it was
falsified**, not because more construction was scheduled.

---

# 1. Dataset facts frozen before signals

The current public dataset is `ds005385` v1.0.3, BIDS 1.9.0, CC0. Its public
GitHub mirror is frozen here at commit:

`6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12`

The study contains 608 participants aged 20–70 years. Resting EEG was recorded
for approximately three minutes in eyes-closed and eyes-open conditions both
before and after a roughly two-hour cognitive block; a subset has an
approximately five-year follow-up.

The primary acquisition metadata are:

- BrainProducts BrainAmp DC;
- EasyCap actiCAP 64;
- 64 EEG channels;
- FCz online reference;
- AFz ground;
- 1000 Hz sampling;
- online low-pass 250 Hz;
- DC high-pass;
- 50 Hz mains;
- nominal file duration 184 s;
- no dedicated EOG channels.

The exact frozen 64-channel order is stored in `channels_64.txt`.

A spot check of `sub-001`, `sub-021`, `sub-043`, and `sub-044` showed the exact
same 64-channel list and identical channel-file Git SHA
`bd9d0e21233ecae20eaeef3978c0f9ed5bbdddad`. The preprocessing code still
requires the exact frozen list for **every** file and fails otherwise.

## Crucial spatial-metadata limitation

The public dataset tree contains no subject-specific `electrodes.tsv` and no
`coordsystem.json`.

Therefore individual electrode digitization and subject-specific EEG/MRI
coregistration are not available from this dataset.

The confirmatory observation model is consequently a **template-head forward
model**, not individual source localization. This is a limitation to report,
not a nuisance to hide.

---

# 2. Development/holdout split is frozen

The historical SMM work had already exposed signals from `sub-001..043`.
Consequently:

\[
\boxed{\text{development}=001\ldots043}
\]

and, conservatively,

\[
\boxed{\text{holdout}=044\ldots608.}
\]

`sub-044` stays holdout even though a current valid public recording exists.
Its historical zero-byte local copy is irrelevant to the present split.

## Metadata-defined development primary set

The dataset itself defines `late_ses1` as indicating late triggers and probable
non-continuity.

Before signal inspection, five of the 43 development subjects are known to
have nonzero `late_ses1`:

| Subject | late_ses1 | Role |
|---|---:|---|
| sub-008 | 1901 | sensitivity only |
| sub-009 | 2080 | sensitivity only |
| sub-013 | 1364 | sensitivity only |
| sub-027 | 15 | sensitivity only |
| sub-037 | 445 | sensitivity only |

Thus the development population is:

\[
\boxed{n_{dev,primary}=38}
\]

plus five predeclared sensitivity subjects.

The confirmatory holdout has:

\[
\boxed{n_{holdout}=565.}
\]

The full machine-readable split is `subject_split.csv`.

No subject may move between these categories based on EEG results.

---

# 3. Primary recording is frozen

The primary recording for both development and holdout is:

\[
\boxed{\text{ses-1 / EyesClosed / acq-pre}.}
\]

The replication order is fixed as:

1. session-1 EyesOpen pre;
2. session-1 EyesClosed post;
3. session-1 EyesOpen post;
4. session-2 EyesClosed pre;
5. remaining session-2 conditions.

The model architecture cannot be retuned between these conditions.

---

# 4. The EEG forward model is fixed scientifically

The confirmatory forward model will use:

- MNE-Python >= 1.13.2;
- the `fsaverage_1005` anatomically shaped 10-05 montage;
- `fsaverage` adult template MRI;
- official `fsaverage-ico-5-src.fif` cortical source space;
- official `fsaverage-5120-5120-5120-bem-sol.fif` three-layer BEM;
- fixed surface-normal source orientation;
- Desikan–Killiany `aparc` cortex, 68 regions;
- explicit common-average reference.

For each DK parcel, the regional lead field is defined as the cortical
surface-area-weighted integral of the fixed-normal vertex lead fields.
Only one global normalization by median regional area is applied, so relative
regional surface area is retained.

The resulting regional operator is

\[
L_{DK}\in\mathbb R^{64\times68}.
\]

The reference operator is explicit:

\[
P=I-\frac{1}{64}\mathbf1\mathbf1^\top,
\qquad
L\leftarrow PL.
\]

## Frozen observable subspace

Before any EEG is seen, the lead field is decomposed:

\[
L=U\Sigma V^\top.
\]

The primary likelihood operates in the fixed first 20 left-singular-vector
sensor subspace:

\[
\boxed{y_{20}=U_{20}^\top y.}
\]

This is **not** data-driven PCA. `U20` is determined solely by the frozen
forward model and is therefore independent of every EEG amplitude and every
model fit.

This reduces redundant sensor dimensions while retaining the best observable
subspace of the template forward operator.

## Current-runtime status

The BEM design is frozen, but the official `fsaverage` BEM/source-space files
are not cached in this runtime and its Python process cannot resolve the
external asset host.

Therefore:

\[
\boxed{\text{NO EEG SIGNAL MAY BE OPENED YET}.}
\]

The only remaining gate before data is a **file/resource gate**:
`build_fsaverage_bem_ds005385.py` must run successfully with the official
assets, save the 64x68 lead field and `U20`, and append their SHA-256 hashes to
the lock.

This is not an unresolved modeling choice.

---

# 5. Preprocessing is frozen and fail-closed

The entire executable procedure is in `preprocess_ds005385.py`.

## 5.1 Input invariants

Every primary file must have:

- exactly the frozen 64 channels in the frozen order;
- 1000-Hz raw sampling;
- at least 180 s before edge cropping;
- the `fsaverage_1005` montage available.

Unexpected metadata are errors, not opportunities for ad-hoc correction.

## 5.2 Edge crop and resampling

Crop 2 s from each end, then resample to 250 Hz.

For a nominal 184-s recording this leaves approximately 180 s.

## 5.3 EDF physical-scale sanity test

The dataset README explicitly warns that EDF physical max/min header values may
be invalid.

Therefore the pipeline **never guesses a rescaling factor**.

After MNE conversion to volts, values are checked in microvolts. The recording
fails preprocessing if either:

\[
\operatorname{median}_c RMS_c <0.1\ \mu V,
\]

or

\[
\operatorname{median}_c RMS_c >500\ \mu V,
\]

or

\[
P_{99.9}(|x|)>5000\ \mu V.
\]

A failed recording goes to explicit EDF-level investigation; it is not
multiplied by powers of ten until it looks plausible.

## 5.4 Global bad channels

Bad-channel detection uses PyPREP 0.9.0 `NoisyChannels` with seed 97,
RANSAC and correlation enabled on a 1–45 Hz copy.

The underlying frozen RANSAC defaults include 50 random samples, 25% sample
proportion, correlation threshold 0.75, bad-window fraction 0.4, and 5-s
windows.

Primary QC fails if:

\[
\boxed{N_{bad}>10.}
\]

No result-dependent manual channel deletion is permitted in the primary run.

## 5.5 ICA / ICLabel

Because there are no dedicated EOG channels, artifact classification is based
on ICA topology/spectrum/autocorrelation rather than an EOG regression channel.

The ICA stream is:

- 1–100 Hz;
- common-average reference;
- extended Infomax;
- seed 97;
- rank-respecting number of components;
- ICLabel 0.9.0.

Components labeled as one of

- muscle artifact;
- eye blink;
- heart beat;
- line noise;
- channel noise

are removed **only** if the predicted class probability is at least 0.80.

`brain` and `other` are retained.

If more than 20% of components would be removed, that recording receives a
QC flag; this threshold cannot be loosened because too many subjects fail.

## 5.6 Analysis stream

The primary analysis signal is:

- 0.5–45 Hz zero-phase FIR;
- common average reference;
- frozen ICA exclusions;
- interpolation of globally bad channels;
- common average reference recomputed after interpolation.

No 50-Hz notch is applied because the primary analysis is already low-passed
below 50 Hz.

## 5.7 Epochs

Use non-overlapping 4-s epochs.

Hard epoch rejection occurs if:

\[
\max_c PTP_c>250\ \mu V,
\]

or if more than 10% of channels have

\[
PTP_c>150\ \mu V.
\]

In addition, an epoch is rejected if either median-channel PTP or
median-channel 30–45-Hz RMS exceeds the across-epoch median by more than
6 robust MAD units.

Primary inclusion requires:

\[
\boxed{\ge30\text{ clean 4-s epochs}=\ge120\text{ s}.}
\]

For slow persistence/metastability analyses, rejected epochs create breaks.
Disjoint retained segments are **never stitched together** as if time were
continuous.

---

# 6. The primary endpoint is not a cherry-picked EEG band

The primary empirical object is the multivariate cross-spectral density (CSD),
not alpha power, theta coherence, or any thresholded “rare coherence” statistic.

The primary empirical CSD is estimated by multitaper over:

\[
1\text{–}40\text{ Hz}
\]

with 1-Hz frequency bins, then projected into the frozen `U20` observable
subspace.

The old 4.65% coherence result, its percentile threshold, and the old
“critical scale” do not enter this analysis at all.

---

# 7. Secondary mechanistic observables are frozen

Secondary analyses include:

1. log-PSD from 1–40 Hz;
2. debiased squared wPLI;
3. imaginary coherency;
4. standard bands:
   - delta 1–4 Hz;
   - theta 4–8 Hz;
   - alpha 8–13 Hz;
   - beta 13–30 Hz;
   - low gamma 30–40 Hz;
5. metastability = SD of the Hilbert-phase order parameter;
6. network persistence = lagged similarity/autocorrelation of vectorized
   leakage-resistant networks across contiguous 4-s windows, up to 20 s;
7. descriptive aperiodic slope 2–30 Hz, excluding 7–14 Hz from its fit.

The order parameter is a **derived statistic**, not a return of Kuramoto as an
independent dynamical layer.

Traveling-wave phase-gradient measures remain exploratory unless their exact
implementation is separately frozen before development results are examined.

Secondary hypothesis families are FDR corrected at q<0.05. They cannot replace
the primary model score after results are known.

---

# 8. Primary model comparison is frozen

The adversarial hierarchy is:

\[
M_0=\text{delayed E-I MPR neuronal network only},
\]

\[
M_1=M_0+\text{generic local one-state slow control},
\]

\[
M_2=M_0+\text{flexible stable generic local two-state slow control},
\]

\[
\boxed{M_3=\text{biophysically constrained K-homeostasis SMM}.}
\]

The microscopic SMM quantities derived upstream are not freely re-estimated per
subject from EEG:

- K-to-QIF coefficients;
- \(q_E,q_I\);
- effective membrane-area prior;
- \(\kappa_e,\kappa_a,\kappa_N\);
- \(D_e,D_g\).

This is essential. Otherwise “SMM” would merely become another flexible slow
latent model.

## Shared fittable network/nuisance quantities

The subject-level fitting space is deliberately small and principally includes:

\[
G_N\in[50,185],
\]

\[
v\in[3,12]\text{ m/s},
\]

plus positive neural-noise scale(s), one global source scale, and a sensor-noise
floor.

The ceiling \(G_N=185\) keeps confirmatory fitting on the stable side of the
pre-data delayed-network transition identified in Step 5.

M1/M2 receive the slow-state parameters needed to be genuine competitors.
Their extra flexibility is paid for by held-out predictive performance rather
than by an informal parameter-count penalty.

---

# 9. Frequency-domain predictive likelihood

For a frozen parameter vector, the linearized delayed stochastic system
produces a model CSD

\[
S_y(\omega;\theta)
=H_y(\omega;\theta)QH_y(\omega;\theta)^*+R.
\]

Delayed structural edges enter as

\[
W_{ij}e^{-i\omega\tau_{ij}}.
\]

The data and all models are projected identically:

\[
S_{20}=U_{20}^*S_{64}U_{20}.
\]

The primary score is the complex multivariate Whittle / complex-Wishart score,
up to model-independent constants:

\[
\boxed{
\ell_M
=-\sum_f\nu_f\left[
\log|S_M(f)|+
\operatorname{tr}\left(S_M(f)^{-1}\hat S(f)\right)
\right].
}
\]

The same eigenvalue regularization rule is applied to every model and empirical
CSD.

Within each recording, predictive evaluation is two-block cross-validation:

- fit first half → score second half;
- fit second half → score first half;
- average held-out score.

This prevents a slow model from winning solely by fitting one realization more
closely in sample.

Executable statistical primitives are in `statistical_lock.py`.

---

# 10. The confirmatory success criterion is frozen

For each holdout subject define:

\[
\Delta_i=
ELPD_i(M_3)-ELPD_i(M_2).
\]

The primary group quantity is the mean paired subject contrast.

Two inferential conditions must both hold:

1. a 10,000-resample subject bootstrap 95% CI has lower bound > 0;
2. a one-sided paired sign-flip test with 100,000 permutations gives p<0.05.

Seeds are fixed to 97.

Therefore:

\[
\boxed{
\text{specific SMM predictive success}
\iff
CI_{low}>0\ \land\ p_{flip}<0.05.
}
\]

The fraction of subjects with \(\Delta_i>0\) is also reported.

Most importantly:

\[
\boxed{M_3>M_0\text{ alone is not success}.}
\]

If the SMM beats neural-only but does not beat the matched generic two-state
slow field, the result supports slow control but not the SMM-specific
physiological constraints.

Even if M3 beats M2, the manuscript must say that EEG **favors a
physiologically constrained SMM conditional on the upstream biology**. EEG does
not directly observe astrocytes.

---

# 11. Replication criterion

After the primary holdout result, the exact same architecture is applied
without mechanism retuning to:

- session-1 EyesOpen pre;
- session-2 EyesClosed pre where available;
- then the post-cognitive-block conditions.

A particularly strong result would be the same sign of M3-vs-M2 predictive
advantage across eyes-open/closed and longitudinal data.

Failure to replicate is reported, not used as a reason to redesign M3 and
rerun the primary holdout.

---

# 12. What may and may not change after development

## May be frozen once after the 38-subject primary development run

Only implementation-level choices that cannot reasonably be fixed without
executing the optimizer may be set once, for example:

- optimizer choice among prespecified reasonable alternatives;
- numerical integration/linear-solver tolerance;
- number of optimizer starts;
- convergence stopping rule;
- CSD numerical floor if the frozen value proves machine-pathological rather
  than scientifically poor.

Any such change must be logged with the complete development result and frozen
before `sub-044` is opened.

## May not change because development results are disappointing

- the biological mechanism;
- K-to-QIF mapping;
- glial topology;
- primary condition;
- development/holdout boundary;
- M2 comparator strength;
- primary M3-vs-M2 contrast;
- model-success criterion;
- primary frequency range;
- choice of a favorable EEG band;
- exclusion thresholds to improve the effect.

---

# 13. Why this is now a finite scientific experiment

Earlier SMM iterations mixed model invention and empirical search. That is what
made the project feel potentially endless: every problem could be answered by
another modeling choice.

That phase is now closed.

The reconstructed chain is fixed:

\[
\boxed{
\text{single-neuron K physiology}
\to
\text{QIF/MPR population dynamics}
\to
\text{local astroglial homeostasis}
\to
\text{delayed neuronal connectome}
\to
\text{pyramidal source currents}
\to
\text{EEG forward model}.
}
\]

The next unknown is no longer “what equation should SMM have?”

It is:

\[
\boxed{
\text{Does this frozen model predict unseen human EEG better than its strongest
matched slow-control null?}
}
\]

That question has a yes/no empirical answer.

---

# 14. Step-5B verdict

## Scientific lock

\[
\boxed{\textbf{PASS — FROZEN}}
\]

## Raw EEG

\[
\boxed{\textbf{NOT OPENED}}
\]

## Template BEM asset

\[
\boxed{\textbf{PENDING EXTERNAL RESOURCE ONLY}}
\]

The current runtime cannot retrieve the official fsaverage BEM assets. The
scientific design of the BEM is already fixed; this does not justify opening
EEG under the older spherical development model.

## Next action

Run `build_fsaverage_bem_ds005385.py`, append its hashes to this lock, then run
the frozen preprocessing on the 38 primary development subjects.

At that point the project is no longer reconstructing SMM.

It is testing it.