# 最终交付清单

## 课程要求对照

| 课程要求 | 项目实现位置 |
|---|---|
| 使用 exp 或 step 构建 CIFAR-100-LT | `src/longtail/data.py`, `scripts/plot_dataset_distribution.py`, `configs/exp_ir100.yaml` |
| 传统机器学习方法 | `src/longtail/train_classical.py`，HOG + PCA + Linear SVM |
| 神经网络方法 | `src/longtail/train_nn.py`, `src/longtail/models.py`，ResNet-32 + CE |
| 训练集与测试集表现 | `metrics.json`, `results/summary.csv`, `scripts/aggregate_results.py` |
| 平衡测试集 | `build_cifar100lt(...)[1]`，原始 CIFAR-100 test |
| 同分布不平衡测试集 | `build_cifar100lt(...)[2]`，按训练协议从 test 抽样 |
| 至少一种性能提升策略 | `src/longtail/losses.py`，LDAM-DRW |
| 数据预处理 | `src/longtail/data.py`, `report/report.md` |
| 模型设置 | `configs/*.yaml`, `report/report.md` |
| 评价指标 | `src/longtail/metrics.py` |
| 实验结论 | `report/report.md`, `report/机器学习课程作业_图像长尾分类实验报告.docx` |
| 类别分布柱状图/折线图 | `report/figures/generated/distribution_*_ir100.png` |
| 不同类别准确率 | `per_class_*.csv`, `report/figures/generated/per_class_*.png` |
| 多种评价和可视化 | `scripts/aggregate_results.py`, `report/figures/generated/` |
| 数据平衡/权重平衡 | class-balanced sampler, class-balanced CE/Focal, LDAM-DRW |

## 当前已实际验证

在当前无 PyTorch/torchvision 的本机环境中，已实际通过：

```powershell
$env:PYTHONPATH="$PWD\src"
python -m py_compile src\longtail\data.py src\longtail\metrics.py src\longtail\train_classical.py src\longtail\train_nn.py src\longtail\losses.py src\longtail\models.py scripts\aggregate_results.py scripts\plot_dataset_distribution.py scripts\build_report_docx.py
python -m pytest tests\test_smoke_pipeline.py -q
python scripts\plot_dataset_distribution.py --config configs\smoke_synthetic_ir100.yaml --out_dir report\figures\generated --all_protocols
python -m longtail.train_classical --config configs\smoke_synthetic_ir100.yaml
python scripts\aggregate_results.py --results_dir results --out results\summary.csv --figures_dir report\figures\generated
python scripts\build_report_docx.py
```

## 需要 GPU/PyTorch 环境运行的正式实验

```powershell
pip install -r requirements-torch.txt
.\scripts\run_full_experiments.ps1
```

完整实验完成后重新运行：

```powershell
python scripts\aggregate_results.py --results_dir results --out results\summary.csv --figures_dir report\figures\generated
python scripts\build_report_docx.py
```

## 交付文件

- `README.md`：项目说明与复现命令。
- `report/report.md`：中文实验报告源文件。
- `report/机器学习课程作业_图像长尾分类实验报告.docx`：正式 Word 报告。
- `results/summary.csv`：当前已验证结果汇总。
- `report/figures/generated/`：自动生成图表。
- `configs/`：正式实验与 smoke 实验配置。
- `scripts/run_full_experiments.ps1`：完整实验一键脚本。
- `scripts/run_smoke_pipeline.ps1`：当前环境快速验证脚本。
