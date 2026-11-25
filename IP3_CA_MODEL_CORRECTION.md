# Mathematical correction: two-field IP3/Ca versus telegraph approximation

**Scope:** The correction concerns the legacy calcium/IP3 branch in `src/smm/glia.py`, **not** the later SMM-K extracellular potassium model.

The historical repository presented a damped-wave model as derived from

```
c_t = -alpha*c + beta*i
i_t = gamma*c - delta*i + D*Laplacian(i)
```

Exact elimination gives

```
c_tt + (alpha+delta)*c_t + (alpha*delta-beta*gamma)*c
 - alpha*D*Laplacian(c) - D*Laplacian(c_t) = 0.
```

The mixed derivative `-D*Laplacian(c_t)` cannot generally be discarded. Even if discarded, the coefficient of `-Laplacian(c)` is `alpha*D`, **not** the historically implemented `D*(delta-alpha)/2`.

For a spatial mode with squared wavenumber `k2`, the exact generator is

```
[[-alpha, beta], [gamma, -(delta+D*k2)]]
```

and its characteristic discriminant is `(alpha-delta-D*k2)**2 + 4*beta*gamma`. For `beta*gamma >= 0`, it is nonnegative, so the linear cooperative subsystem has real modal eigenvalues rather than oscillatory wave modes. The historical telegraph PDE can be studied as a phenomenological model, but its wave speed is **not derived** from the proposed two-field microscopic equations.

## Compatibility and scientific interpretation

* `src/smm/ip3_ca_exact.py` implements the exact modal generator, eigenvalues, and eliminated characteristic polynomial. All rates are in `s^-1`, `D` in `um^2/s`, and `k2` in `um^-2`. The rates are illustrative, not experimentally calibrated.
* The historical solver and parameter mapping are retained to reproduce legacy numerical studies; calls to its micro-to-telegraph function now issue a warning rather than silently claiming a mechanistic derivation.
* Existing legacy tests check numerical functionality, not validity of the microscopic wave interpretation.
* This fix does **not** establish or refute the separate potassium-based SMM-K mechanism. SMM-K requires its own physiological parameter calibration, alternative-mechanism controls, and direct experimental validation.

Run `python -m pytest tests/test_ip3_ca_exact.py -q` (11 tests verified locally). The exact two-variable model does not, by itself, supply the physically justified glial wave equation claimed in older documentation.
