#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/src"

python scripts/plot_dataset_distribution.py --config configs/smoke_synthetic_ir100.yaml --out_dir report/figures/generated --all_protocols
python -m longtail.train_classical --config configs/smoke_synthetic_ir100.yaml
python scripts/aggregate_results.py --results_dir results --out results/summary.csv --figures_dir report/figures/generated
