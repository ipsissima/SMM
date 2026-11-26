# Short-time distinguishability of glial versus extracellular potassium transport

**Status:** A rigorously derived statement for the canonical linear two-compartment SMM-K reduction, **not** a statement about the full electrodiffusive (EDNM) equations or a biological experiment.

For extracellular potassium perturbation k and glial excess load a,

```
phi_e * k_dot = u - (kappa_e+kappa_N)*k + kappa_a*a - D_e*L_e*k
phi_a * a_dot = kappa_e*k - kappa_a*a - D_g*L_g*a
```

Assume k(0)=a(0)=0, time-independent focal source u, and bounded linear spatial operators L_e and L_g. Neither operator need commute with the other. As t -> 0+,

```
d k(t)/d D_e = - (L_e @ u) * t^2 / (2 * phi_e^2) + O(t^3)

d k(t)/d D_g = - kappa_a*kappa_e*(L_g @ u) * t^4
                / (24 * phi_e^2 * phi_a^2) + O(t^5).
```

If a neuronal state n obeys n_dot=A_n*n + B_n*k, n(0)=0, with no direct dependence on D_g, its transport sensitivity is

```
d n(t)/d D_g = - kappa_a*kappa_e * B_n@(L_g@u) * t^5
                / (120*phi_e^2*phi_a^2) + O(t^6).
```

**Proof:** For x=(k,a), generator F=[A,B;C,D] and source b=(u/phi_e,0), use x(t)=sum_(j>=0) t^(j+1)F^j b/(j+1)!. The extracellular spatial coefficient first appears in the upper block of F b. The first glial spatial contribution to extracellular potassium must traverse k -> a -> spatial glial transport -> a -> k, so it first appears in the upper block of F^3 b, with product B (dD/dD_g) C b_k. The downstream neuronal state adds an integration, shifting the leading term to order t^5.

**Independent numerical check:** A two-node model evaluated using exact matrix-exponential Frechet derivatives gave ratios of computed / leading predicted coefficients at t=1 ms of 0.99503 (glial, fourth order) and 0.99294 (extracellular, second order). A downstream simple neuronal observer gave ratio 0.98679 for the fifth-order glial term at t=2 ms.

This relative-order result helps explain why a short potassium perturbation can show large extracellular spatial propagation while remaining almost uninformative about glial transport in the reduced model. It must not be generalized without analysis to full source KNP electrochemical equations, nonzero glial initial loads, direct astroglial forcing, and longer nonlinear responses.
