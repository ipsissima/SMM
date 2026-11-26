# Reproducing the foundational calculations

Python 3 with NumPy, SciPy and Matplotlib. Run from this directory with the original source archives in inputs/ (or the repository root, as supported by matched_diagnostics.py).

```sh
python -m venv .venv
. .venv/bin/activate
pip install numpy scipy matplotlib
OPENBLAS_NUM_THREADS=1 python matched_diagnostics.py
OPENBLAS_NUM_THREADS=1 python shared_path_test.py
```

[results.json](results.json) and [shared_path_results.json](shared_path_results.json) are the archived numerical outputs. The passive rates, rest fractions, stimulation amplitude, sampling times and chord weights are explicit in the scripts and manuscript. The noise model is illustrative, not empirical. The scripts recreate the passive analysis and use the original EDNM archives for the matched comparison.

Retrieve the original archives from the following GitHub Actions runs, retaining their original contents and naming the ZIP files as indicated in the scripts:

* [45847600616](https://github.com/ipsissima/SMM/actions/runs/45847600616): EDNM7_GRAPH_FINE_path.zip (15131038799), EDNM7_GRAPH_FINE_shortcut.zip (15132425485).
* [45839123584](https://github.com/ipsissima/SMM/actions/runs/45839123584): EDNM_SECONDS_native.zip (15125194998), EDNM_SECONDS_native_convergence.zip (15127223038).
* [45840811721](https://github.com/ipsissima/SMM/actions/runs/45840811721): EDNM_SECONDS_glial_off.zip (15128570093), EDNM_SECONDS_glial_off_fine.zip (15128996238).
* [45859252351](https://github.com/ipsissima/SMM/actions/runs/45859252351): pump clamp and high-load source reproduction.

The exact full-source drivers are pinned in [the validation status](../docs/VALIDATION.md). Run the corresponding workflows to regenerate arrays if artifacts expire. For the current accounting, extract baseline/data.npz and stimulated/data.npz and use `python ednm_voltage_audit.py compare --help`. A permanent raw-array DOI deposit is outstanding. The archived high-load workflow encountered a NumPy integer JSON serialization error after saving arrays; the scientific checks were performed on those saved arrays, not inferred from an exit status.

The legacy scripts at repository root concern earlier SMM realizations and should not be confused with this package.
