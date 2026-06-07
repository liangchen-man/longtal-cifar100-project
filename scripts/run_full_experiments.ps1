$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "$PWD\src"

python scripts\plot_dataset_distribution.py --config configs\exp_ir100.yaml --out_dir report\figures\generated --all_protocols

python -m longtail.train_classical --config configs\exp_ir100.yaml

python -m longtail.train_nn --config configs\ce_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\ldam_drw_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\balanced_softmax_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\logit_adjusted_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\class_balanced_focal_resnet32_ir100.yaml

python scripts\aggregate_results.py --results_dir results --out results\summary.csv --figures_dir report\figures\generated
