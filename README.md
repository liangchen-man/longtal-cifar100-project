# CIFAR-100-LT 图像长尾分类实验项目

这是面向“题目 1 图像长尾分类实验”的完整工程交付版，覆盖课程基础要求，并扩展了长尾识别常用的论文级评估与改进方法。

## 已实现内容

- `exp` 与 `step` 两种 CIFAR-100-LT 不平衡构建方式。
- 传统机器学习方法：`HOG + PCA + Linear SVM`。
- 神经网络方法：CIFAR 风格 `ResNet-32 + Cross Entropy`。
- 长尾改进策略：`LDAM-DRW`。
- 额外增强方法：`Balanced Softmax`、`Logit Adjustment`、`Class-Balanced Focal`、class-balanced sampling。
- 平衡测试集与同分布不平衡测试集双测试协议。
- Top-1、Top-5、Macro-F1、Balanced Acc、Many/Medium/Few Acc、ECE、逐类准确率、混淆矩阵、reliability diagram、LDAM scale 消融图。
- 结果聚合、分布图、整体对比图、逐类准确率图、训练历史曲线。
- 中文实验报告源文件与 DOCX 生成脚本。

## 当前环境说明

本项目已在 `.conda-cuda` 环境中完成 CIFAR-100-LT `exp IR=100` 正式实验。该环境包含 `torch 2.12.0+cu126`、`torchvision 0.27.0+cu126`，并可使用 CUDA；数据位于 `data/cifar-100-python`。

主要结果如下：

- `ResNet-32 + CE`：平衡测试 Top-1 40.66%，不平衡测试 Top-1 64.71%，平衡测试 Few Acc 9.63%。
- `Balanced Softmax / Logit Adjustment τ=1`：平衡测试 Top-1 44.89%，Macro-F1 43.93%，Few Acc 27.27%，是本轮补跑中综合最好的改进策略。
- 原始 `LDAM-DRW (s=30)`：本设置下未收敛，平衡测试 Top-1 1.00%。
- 调参版 `LDAM-DRW (s=1)`：平衡测试 Top-1 44.60%，Macro-F1 42.28%，Few Acc 19.87%；相对 CE 更好地改善尾部类别。
- `Class-Balanced Focal`：平衡测试 Top-1 21.55%，Few Acc 19.13%，说明强重加权虽然能照顾尾部，但会显著损害整体分类。
- 传统 `HOG+PCA+LinearSVM`：平衡测试 Top-1 12.63%，作为课程传统方法基线。

完整指标见 `results/summary.csv` 与 `results/summary.md`，报告见 `report/report.md` 和 `report/机器学习课程作业_图像长尾分类实验报告.docx`。

## 快速验证

Windows PowerShell：

```powershell
.\scripts\run_smoke_pipeline.ps1
```

等价手动命令：

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts\plot_dataset_distribution.py --config configs\smoke_synthetic_ir100.yaml --out_dir report\figures\generated --all_protocols
python -m longtail.train_classical --config configs\smoke_synthetic_ir100.yaml
python scripts\aggregate_results.py --results_dir results --out results\summary.csv --figures_dir report\figures\generated
```

已生成的 smoke 与正式实验结果位于：

- `results/summary.csv`
- `results/smoke_synthetic_ir100/classical_hog_svm/metrics.json`
- `results/ce_resnet32_ir100/metrics.json`
- `results/ldam_drw_resnet32_ir100/metrics.json`
- `results/ldam_drw_resnet32_ir100_s1/metrics.json`
- `report/figures/generated/`

## 完整 CIFAR-100-LT 复现

安装依赖：

```bash
pip install -e ".[torch,dev]"
```

运行完整实验：

```powershell
.\scripts\run_full_experiments.ps1
```

或逐个运行：

```powershell
$env:PYTHONPATH="$PWD\src"
python -m longtail.train_classical --config configs\exp_ir100.yaml
python -m longtail.train_nn --config configs\ce_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\ldam_drw_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\balanced_softmax_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\logit_adjusted_resnet32_ir100.yaml
python -m longtail.train_nn --config configs\class_balanced_focal_resnet32_ir100.yaml
python scripts\aggregate_results.py --results_dir results --out results\summary.csv --figures_dir report\figures\generated
```

## 项目结构

```text
configs/                         实验配置
scripts/                         分布图、聚合、完整/快速运行脚本
src/longtail/                    数据、模型、损失、指标、训练代码
tests/                           smoke 回归测试
report/report.md                 中文实验报告源文件
report/figures/generated/        自动生成图表
results/                         实验结果
original/LDAM-DRW-master/        用户提供的 LDAM-DRW 原始代码参考
```

## 核心文件

- `src/longtail/data.py`：CIFAR-100-LT 与 synthetic smoke 数据构建。
- `src/longtail/train_classical.py`：HOG+PCA+LinearSVM。
- `src/longtail/train_nn.py`：ResNet 训练、验证、checkpoint、history。
- `src/longtail/losses.py`：CE、LDAM-DRW、Balanced Softmax、Logit Adjustment、CB-Focal。
- `src/longtail/metrics.py`：完整长尾分类指标。
- `scripts/aggregate_results.py`：汇总 `metrics.json` 并生成表格和图。

## 报告

Markdown 报告：

```text
report/report.md
```

生成 DOCX：

```powershell
python scripts\build_report_docx.py
```

## 重要实验原则

长尾分类不能只看 overall Top-1。不平衡测试集上的 overall accuracy 可能因为头部类占比高而看似更好；平衡测试集上的 Macro-F1、Balanced Acc、Few Acc 和逐类准确率更能反映模型是否真正解决尾部类别问题。
