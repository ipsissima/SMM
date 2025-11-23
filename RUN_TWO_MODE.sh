#!/usr/bin/env bash
set -euo pipefail

# Smoke run for quick validation
python supplementary_code/two_mode_fitting.py \
  --input PSD_with_Coherence.csv \
  --outdir supplementary_code/two_mode_results_smoke \
  --B 100 \
  --seed 1234 \
  --threshold 0.05 \
  --assume-constant \
  --smoke

# Full run
python supplementary_code/two_mode_fitting.py \
  --input PSD_with_Coherence.csv \
  --outdir supplementary_code/two_mode_results \
  --B 1000 \
  --seed 1234 \
  --threshold 0.05 \
  --assume-constant
