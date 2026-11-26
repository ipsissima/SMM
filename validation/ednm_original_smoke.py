"""Bounded unmodified EDNM full-solver smoke test. Not a scientific validation.

Requires Sætra-Mori reference checked out at pinned commit, Python 3.10, and
original Numba AOT residual/Jacobian compiled. It runs NO neuronal stimulus.
No original model equations are changed. Uses sealed boundaries for an
ion-conservation diagnostic.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from ednm import model, solver

def ions_inventory(data):
    c = data["c"]
    alpha = data["alpha"]
    # c: (time,layer,ion,domain,unit); alpha: (time,layer,2,unit)
    rest = 1.0 - alpha.sum(axis=2, keepdims=True)
    frac = np.concatenate([alpha, rest], axis=2)
    return np.einsum("tldu,tlidu->ti", frac, c)

def main():
    out = Path(os.environ.get("EDNM_OUTPUT_DIR","ednm-output"))
    out.mkdir(exist_ok=True, parents=True)
    N = 2
    dt = 5e-5
    duration = .002
    m = model.Model(N, 5e-4, 0, 0)
    sp = np.zeros((2,N),dtype=float)
    protocol = {"j_stim":0.,"stim_start":0.,"stim_end":0.,
                "N_stim":0,"spike_train":sp}
    with (out/"solver.log").open("w") as f, contextlib.redirect_stdout(f):
        solver.solve_system(m,str(out)+"/",duration,protocol)
    d = np.load(out/"data.npz")
    for k in ("t","alpha","c","phi","ss"):
        assert np.isfinite(d[k]).all(), k
    b = ions_inventory(d)
    drift_abs = np.abs(b[-1]-b[0])
    drift_relative = drift_abs / np.maximum(np.abs(b[0]),1e-12)
    baseline_ke = float(d["c"][0,0,1,2,0])
    latest_ke = float(d["c"][-1,0,1,2,0])
    result = {"source":"Saetra-Mori EDNM (unmodified)",
        "original_ref":"826c211c70d9027a4acb02202ca91a60de79a15d",
        "kind":"full original implicit solver, 2-unit sealed no-stimulus SMOKE TEST",
        "num_units":N,"dt_seconds":dt,
        "duration_seconds":duration,"n_samples":len(d["t"]),
        "initial_ke_soma_mM":baseline_ke,"final_ke_soma_mM":latest_ke,
        "absolute_ion_inventory_drift":dict(zip(["Na","K","Cl","Ca"],map(float,drift_abs))),
        "relative_ion_inventory_drift":dict(zip(["Na","K","Cl","Ca"],map(float,drift_relative))),
        "runtime_note":"Short smoke test; does not establish stable equilibrium or biophysical validity",
        "numpy_version":np.__version__}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
