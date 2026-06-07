#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/longtail_cifar100_project}"
PY="${PY:-/root/miniconda3/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

mkdir -p logs_remote

run_train() {
  local config="$1"
  local name
  name="$(basename "$config" .yaml)"
  local out_dir
  out_dir="$("$PY" - "$config" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["output_dir"])
PY
)"
  if [[ -f "$out_dir/metrics.json" ]]; then
    echo "[$(date '+%F %T')] skip $name: $out_dir/metrics.json exists"
    return
  fi
  echo "[$(date '+%F %T')] start $name"
  "$PY" src/longtail/train_nn.py --config "$config" 2>&1 | tee "logs_remote/${name}.log"
  echo "[$(date '+%F %T')] done $name"
}

refresh_metrics() {
  local config="$1"
  local name
  name="$(basename "$config" .yaml)"
  local out_dir
  out_dir="$("$PY" - "$config" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["output_dir"])
PY
)"
  if [[ ! -f "$out_dir/best.pt" ]]; then
    echo "[$(date '+%F %T')] skip refresh $name: no best.pt"
    return
  fi
  echo "[$(date '+%F %T')] refresh metrics $name"
  "$PY" scripts/refresh_nn_metrics.py --config "$config" 2>&1 | tee "logs_remote/refresh_${name}.log"
}

echo "[$(date '+%F %T')] remote extra experiments begin"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

run_train configs/balanced_softmax_resnet32_ir100.yaml
run_train configs/logit_adjusted_resnet32_ir100.yaml
run_train configs/class_balanced_focal_resnet32_ir100.yaml

run_train configs/ldam_drw_resnet32_ir100_s0_5_short.yaml
run_train configs/ldam_drw_resnet32_ir100_s1_short.yaml
run_train configs/ldam_drw_resnet32_ir100_s2_short.yaml
run_train configs/ldam_drw_resnet32_ir100_s5_short.yaml
run_train configs/ldam_drw_resnet32_ir100_s10_short.yaml
run_train configs/ldam_drw_resnet32_ir100_s30_short.yaml

refresh_metrics configs/ce_resnet32_ir100.yaml
refresh_metrics configs/ldam_drw_resnet32_ir100.yaml
refresh_metrics configs/ldam_drw_resnet32_ir100_s1.yaml
refresh_metrics configs/balanced_softmax_resnet32_ir100.yaml
refresh_metrics configs/logit_adjusted_resnet32_ir100.yaml
refresh_metrics configs/class_balanced_focal_resnet32_ir100.yaml

"$PY" scripts/aggregate_results.py --results_dir results --out results/summary.csv --figures_dir report/figures/generated 2>&1 | tee logs_remote/aggregate_results.log
"$PY" scripts/plot_diagnostics.py --results_dir results --summary results/summary.csv --figures_dir report/figures/generated 2>&1 | tee logs_remote/plot_diagnostics.log

echo "[$(date '+%F %T')] remote extra experiments complete"
