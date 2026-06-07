# 机器学习课程作业：CIFAR-100-LT 图像长尾分类实验

班级：__________  姓名：__________  学号：__________  日期：2026 年 6 月

## 摘要

本项目围绕“图像长尾分类实验”构建了完整的 CIFAR-100-LT 实验工程。数据层支持 `exp` 与 `step` 两种不平衡构造方式；模型层包含传统机器学习基线 `HOG + PCA + Linear SVM` 与神经网络基线 `ResNet-32 + Cross Entropy`；改进策略实现了 `LDAM-DRW`，并额外正式补跑了 `Balanced Softmax`、`Logit Adjustment`、`Class-Balanced Focal` 等长尾学习常用方法。评估协议同时包含平衡测试集与和训练集同分布的不平衡测试集，并报告 Top-1、Top-5、Macro-F1、Balanced Accuracy、Many/Medium/Few Acc、ECE、逐类准确率、混淆矩阵与 reliability diagram。

本实验已完成 CIFAR-100-LT `exp IR=100` 正式评估：传统 HOG+PCA+LinearSVM、ResNet-32+CE、原始 LDAM-DRW、调参版 LDAM-DRW `s=1`、Balanced Softmax、Logit Adjustment 与 Class-Balanced Focal 均已生成 `metrics.json`、逐类结果、训练历史和对比图。结果显示，CE 在不平衡测试集上 Top-1 达到 64.71%，但平衡测试集 Few Acc 仅 9.63%；Balanced Softmax / Logit Adjustment 在平衡测试集上达到 Top-1 44.89%、Macro-F1 43.93%、Few Acc 27.27%，是本组实验中最有效的长尾改进策略。

## 1. 任务与研究问题

真实视觉数据通常呈现长尾分布：少数头部类别有大量样本，多数尾部类别样本稀缺。普通经验风险最小化模型容易被头部类别主导，表现为整体准确率看似较高，但尾部类别召回率、宏平均指标和类别公平性显著下降。本实验目标是：

1. 使用 `exp` 或 `step` 方式构建 CIFAR-100-LT 不平衡训练集。
2. 使用 1 种传统机器学习方法与 1 种神经网络方法进行分类。
3. 在训练集、平衡测试集、不平衡测试集上报告表现并比较。
4. 针对神经网络方法采用至少一种合理长尾策略提升测试性能。
5. 说明数据预处理、模型设置、评价指标、可视化与实验结论。

## 2. 数据集构建

### 2.1 CIFAR-100-LT 协议

原始 CIFAR-100 训练集包含 100 类，每类 500 张训练图像；测试集每类 100 张。项目按类别索引从头到尾排序，并根据不平衡类型为第 `i` 类分配样本数。

`exp` 长尾构造：

```text
n_i = n_max * (1 / IR) ** (i / (C - 1))
```

其中 `C=100`，`n_max=500`，`IR=100` 时，头部类约 500 张，尾部类约 5 张。

`step` 长尾构造：

```text
前 50 类：n_i = n_max
后 50 类：n_i = n_max / IR
```

测试时同时构建两套集合：

- 平衡测试集：保留 CIFAR-100 原始测试集，每类 100 张，用于衡量类别公平性和泛化性能。
- 不平衡测试集：按训练集同样的 `exp/step` 分布从测试集抽样，用于模拟部署时测试分布也长尾的情况。

### 2.2 数据预处理

传统模型使用未增强图像提取 HOG 特征，避免随机增强破坏手工特征可复现性。神经网络训练使用 CIFAR 常规增强：

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip`
- `ToTensor`
- 按 CIFAR-100 均值方差归一化，均值为 `(0.5071, 0.4867, 0.4408)`，标准差为 `(0.2675, 0.2565, 0.2761)`

项目已生成 `exp` 与 `step` 两种分布图：

- `report/figures/generated/distribution_exp_ir100.png`
- `report/figures/generated/distribution_step_ir100.png`

## 3. 方法

### 3.1 传统机器学习基线：HOG + PCA + Linear SVM

传统方法采用以下流水线：

1. 对 32x32 RGB 图像提取 HOG 特征，方向数为 9，cell 大小为 4x4，block 大小为 2x2。
2. 使用 `StandardScaler` 标准化特征。
3. 使用 PCA 降维，默认保留 256 维；smoke 配置中使用 128 维。
4. 使用 `SGDClassifier(loss="hinge", class_weight="balanced")` 训练线性 SVM。

该方法满足课程对传统机器学习分类器的要求，并提供一个不依赖深度网络表示学习的可解释基线。预期上，它在低层纹理/颜色模式明显的类别上有效，但对 CIFAR-100 的细粒度语义类别能力有限。

### 3.2 神经网络基线：ResNet-32 + Cross Entropy

神经网络基线采用 CIFAR 风格 ResNet-32，即深度满足 `6n+2`，`n=5`。网络包含三个残差 stage，通道数为 16、32、64，最后使用全局平均池化和全连接分类器。训练损失为标准交叉熵：

```text
L_CE = -log softmax(z_y)
```

该基线代表普通经验风险最小化方法。在长尾训练集上，CE 往往使分类边界偏向尾部类别，尾部类召回较差。

### 3.3 改进方法：LDAM-DRW

主改进策略为 LDAM-DRW。LDAM 为每个类别设置类别相关 margin：

```text
m_j ∝ 1 / n_j ** 1/4
```

样本少的尾部类具有更大的 margin，从而迫使模型为尾部类别学习更宽的分类间隔。DRW（Deferred Re-Weighting）在训练前期不加权，使网络先学习通用表示；后期再使用基于 effective number 的类别权重，降低头部类梯度主导。

项目配置中默认 `drw_start_epoch=160`，与 200 epoch CIFAR 实验惯例一致。

### 3.4 额外长尾方法

为了让分析更完整，项目还实现了以下方法：

- Balanced Softmax：将训练集类别先验加入 softmax 分母，显式修正标签分布偏移。
- Logit Adjustment：在 logits 上加入 `tau * log p_y` 的先验调整，常用于长尾分类和校准。
- Class-Balanced Focal：结合 effective number 类权重和 Focal Loss，抑制易分类头部样本的梯度占比。
- Class-Balanced Sampling：按样本所属类别频次构造 weighted sampler，作为数据平衡策略。

## 4. 实验设置

### 4.1 正式 CIFAR-100-LT 实验矩阵

| 方法 | 类型 | 配置文件 | 目的 |
|---|---|---|---|
| HOG+PCA+LinearSVM | 传统机器学习 | `configs/exp_ir100.yaml` | 课程传统模型要求 |
| ResNet-32+CE | 神经网络基线 | `configs/ce_resnet32_ir100.yaml` | 观察长尾偏置 |
| ResNet-32+LDAM-DRW | 主改进方法 | `configs/ldam_drw_resnet32_ir100.yaml` | 提升尾部类别与宏平均指标 |
| ResNet-32+Balanced Softmax | 分布修正 | `configs/balanced_softmax_resnet32_ir100.yaml` | 修正训练先验偏置 |
| ResNet-32+Logit Adjustment | 后验校正 | `configs/logit_adjusted_resnet32_ir100.yaml` | 提升平衡测试集与校准 |
| ResNet-32+CB-Focal | 权重与难例调制 | `configs/class_balanced_focal_resnet32_ir100.yaml` | 降低头部易例主导 |

### 4.2 评价指标

本项目不仅报告 overall accuracy，还重点报告长尾分类更敏感的指标：

- Top-1 / Top-5：分类准确率。
- Macro-F1：对每类同等加权，反映尾部类表现。
- Balanced Accuracy：逐类召回的平均值。
- Many/Medium/Few Acc：按训练样本数分组统计逐类准确率。默认 `many >= 100`，`20 <= medium < 100`，`few < 20`。
- Per-class Accuracy：展示每个类别的准确率曲线。
- Confusion Matrix：分析类别混淆结构。
- ECE：Expected Calibration Error，用于衡量模型置信度校准。

## 5. 正式 CIFAR-100-LT 实验结果

### 5.1 运行环境与执行命令

本次正式实验使用项目内 `.conda-cuda` 环境运行，PyTorch 版本为 `2.12.0+cu126`，`torchvision` 版本为 `0.27.0+cu126`，CUDA 可用。数据使用本地 `data/cifar-100-python`，训练协议为 `exp IR=100`，神经网络训练 200 epoch，学习率里程碑为 160 和 180 epoch。

已实际运行的核心命令如下：

```powershell
.\.conda-cuda\python.exe src\longtail\train_nn.py --config configs\ce_resnet32_ir100.yaml
.\.conda-cuda\python.exe src\longtail\train_nn.py --config configs\ldam_drw_resnet32_ir100.yaml
.\.conda-cuda\python.exe src\longtail\train_nn.py --config configs\ldam_drw_resnet32_ir100_s1.yaml
.\.conda-cuda\python.exe src\longtail\train_nn.py --config configs\balanced_softmax_resnet32_ir100.yaml
.\.conda-cuda\python.exe src\longtail\train_nn.py --config configs\logit_adjusted_resnet32_ir100.yaml
.\.conda-cuda\python.exe src\longtail\train_nn.py --config configs\class_balanced_focal_resnet32_ir100.yaml
.\.conda-cuda\python.exe scripts\aggregate_results.py --results_dir results --out results\summary.csv --figures_dir report\figures\generated
```

生成结果文件：

- `results/ce_resnet32_ir100/metrics.json`
- `results/ldam_drw_resnet32_ir100/metrics.json`
- `results/ldam_drw_resnet32_ir100_s1/metrics.json`
- `results/balanced_softmax_resnet32_ir100/metrics.json`
- `results/logit_adjusted_resnet32_ir100/metrics.json`
- `results/class_balanced_focal_resnet32_ir100/metrics.json`
- `results/summary.csv`
- `report/figures/generated/comparison_top1.png`
- `report/figures/generated/comparison_macro_f1.png`
- `report/figures/generated/comparison_few_acc.png`
- `report/figures/generated/per_class_ce_resnet32_ir100_balanced_test.png`
- `report/figures/generated/per_class_ldam_drw_resnet32_ir100_s1_balanced_test.png`
- `report/figures/generated/confusion_balanced_softmax_resnet32_ir100_balanced_test.png`
- `report/figures/generated/reliability_balanced_softmax_resnet32_ir100_balanced_test.png`
- `report/figures/generated/ldam_scale_ablation.png`

结果汇总如下：

| 方法 | Split | Top-1 | Top-5 | Macro-F1 | Balanced Acc | Medium Acc | Few Acc | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HOG+PCA+LinearSVM | Balanced Test | 12.63 | 30.62 | 10.94 | 12.63 | 11.94 | 1.63 | 87.18 |
| HOG+PCA+LinearSVM | Imbalanced Test | 20.46 | 46.88 | 10.50 | 11.03 | 10.13 | 0.00 | 79.41 |
| ResNet-32+CE | Balanced Test | 40.66 | 69.94 | 36.20 | 40.66 | 40.17 | 9.63 | 32.57 |
| ResNet-32+CE | Imbalanced Test | 64.71 | 86.49 | 39.14 | 39.29 | 39.57 | 7.22 | 16.24 |
| LDAM-DRW 原始 s=30 | Balanced Test | 1.00 | 5.00 | 0.02 | 1.00 | 0.00 | 0.00 | 28.24 |
| LDAM-DRW 原始 s=30 | Imbalanced Test | 2.91 | 12.20 | 0.06 | 1.00 | 0.00 | 0.00 | 26.33 |
| LDAM-DRW 调参 s=1 | Balanced Test | 44.60 | 73.43 | 42.28 | 44.60 | 44.94 | 19.87 | 28.73 |
| LDAM-DRW 调参 s=1 | Imbalanced Test | 63.12 | 84.37 | 41.88 | 43.33 | 44.62 | 17.22 | 16.15 |
| Balanced Softmax | Balanced Test | 44.89 | 74.04 | 43.93 | 44.89 | 44.51 | 27.27 | 22.41 |
| Balanced Softmax | Imbalanced Test | 58.05 | 80.53 | 38.39 | 44.44 | 42.95 | 28.89 | 15.94 |
| Logit Adjustment τ=1 | Balanced Test | 44.89 | 74.04 | 43.93 | 44.89 | 44.51 | 27.27 | 22.41 |
| Logit Adjustment τ=1 | Imbalanced Test | 58.05 | 80.53 | 38.39 | 44.44 | 42.95 | 28.89 | 15.94 |
| Class-Balanced Focal | Balanced Test | 21.55 | 47.15 | 18.55 | 21.55 | 36.60 | 19.13 | 13.39 |
| Class-Balanced Focal | Imbalanced Test | 10.09 | 35.43 | 9.62 | 19.61 | 33.39 | 15.56 | 19.96 |

可以看到，不平衡测试集上的 Top-1 普遍高于平衡测试集，这是因为测试分布同样偏向头部类别；但长尾分类更应关注平衡测试集、Macro-F1 和 Few Acc。CE 的不平衡测试 Top-1 最高，为 64.71%，但平衡测试 Few Acc 只有 9.63%。调参后的 LDAM-DRW 在平衡测试 Top-1、Macro-F1 与 Few Acc 上均优于 CE；进一步补跑的 Balanced Softmax / Logit Adjustment τ=1 将平衡测试 Top-1 提升到 44.89%、Macro-F1 提升到 43.93%、Few Acc 提升到 27.27%，是本实验中尾部类别改善最明显的策略。Class-Balanced Focal 的 Few Acc 高于 CE，但 overall Top-1 明显下降，说明过强的类别重加权和难例调制会牺牲大量头部与整体性能。

### 5.3 LDAM-DRW 诊断说明

原始 `configs/ldam_drw_resnet32_ir100.yaml` 采用论文常见的 `s=30` scale，但在本工程的 ResNet-32 与学习率设置下未能收敛，训练损失长期处于很高尺度，最终平衡测试 Top-1 仅 1.00%。为判断问题是否来自实现错误还是尺度超参，本项目额外运行了 20 epoch 诊断配置 `configs/ldam_drw_resnet32_ir100_diag.yaml`，仅将 `s` 改为 1 后训练立即正常下降，20 epoch 平衡测试 Top-1 达 29.24%。因此正式对照中保留原始失败结果，同时新增 `configs/ldam_drw_resnet32_ir100_s1.yaml` 作为“调参版 LDAM-DRW”。

进一步补跑的 30 epoch scale 消融如下：

| LDAM scale | Balanced Top-1 | Macro-F1 | Few Acc | ECE |
|---:|---:|---:|---:|---:|
| 0.5 | 34.50 | 29.31 | 9.20 | 36.63 |
| 1 | 35.90 | 31.34 | 10.73 | 12.97 |
| 2 | 34.94 | 30.69 | 12.40 | 12.12 |
| 5 | 4.44 | 1.45 | 0.03 | 2.91 |
| 10 | 1.00 | 0.02 | 0.00 | 0.42 |
| 30 | 1.01 | 0.04 | 0.00 | 21.23 |

该结果表明，`s=0.5/1/2` 在短跑阶段能够正常学习，而 `s>=5` 很快退化到接近随机或单类预测。也就是说，原始 `s=30` 的失败主要来自当前实现、学习率和模型容量组合下的 logit scale 过大，而非 LDAM margin 思想本身失效。对应折线图见 `report/figures/generated/ldam_scale_ablation.png`。

## 6. 实验分析

1. HOG+PCA+LinearSVM 明显弱于 ResNet-32：平衡测试 Top-1 只有 12.63%，说明手工纹理特征难以覆盖 CIFAR-100 的细粒度语义差异。
2. ResNet-32+CE 在不平衡测试集上 Top-1 很高，但 Few Acc 偏低，说明模型主要受益于头部类频次，并没有充分解决尾部类别。
3. 原始 `s=30` LDAM-DRW 在本设置中未收敛，不能作为有效改进结论；这也说明长尾损失函数的尺度超参和学习率需要联合诊断。
4. 调参后的 `s=1` LDAM-DRW 更符合长尾目标：牺牲少量不平衡测试 overall Top-1，换来平衡测试、Macro-F1、Balanced Acc 和 Few Acc 的提升。
5. Balanced Softmax / Logit Adjustment τ=1 是本实验中综合最好的改进策略，平衡测试 Top-1 为 44.89%，Few Acc 为 27.27%。二者结果完全一致并非偶然，因为当前实现中两者都等价于在训练 logits 中加入 `log p_y` 类别先验。
6. Class-Balanced Focal 将 Few Acc 提高到 19.13%，但平衡测试 Top-1 只有 21.55%，说明强重加权和 focal 调制可能过度牺牲头部类与整体判别能力。
7. 逐类准确率、混淆矩阵和 reliability diagram 显示，CE 更偏向头部类别；Balanced Softmax 和 LDAM-DRW(s=1) 能让尾部类出现更连续的非零准确率，但校准误差仍有进一步改进空间。

## 7. 复现方式

### 7.1 当前环境 smoke 复现

```powershell
.\scripts\run_smoke_pipeline.ps1
```

### 7.2 完整 CIFAR-100-LT 实验

在安装 `torch`、`torchvision` 且可下载或本地已有 CIFAR-100 数据后运行：

```powershell
.\scripts\run_full_experiments.ps1
```

或逐个执行：

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

## 8. 结论

本项目已经完成课程基础要求并扩展为完整长尾识别实验框架：支持 `exp/step` CIFAR-100-LT 构建、传统机器学习与神经网络分类、平衡/不平衡双测试协议、LDAM-DRW 改进策略、多个前沿长尾对照方法、逐类和分组指标、混淆矩阵、reliability diagram、可视化与报告生成。本次已完成 CIFAR-100-LT `exp IR=100` 正式训练与评估，其中 CE 平衡测试 Top-1 为 40.66%，调参版 LDAM-DRW 平衡测试 Top-1 为 44.60%，Balanced Softmax / Logit Adjustment τ=1 进一步达到 44.89%，Few Acc 从 CE 的 9.63% 提升到 27.27%。

从方法论上，长尾分类不能只看 overall accuracy。不平衡测试集上的 CE Top-1 最高，但它的尾部类别准确率较低；Balanced Softmax、Logit Adjustment 和调参版 LDAM-DRW 在平衡测试集、Macro-F1、Balanced Acc 与 Few Acc 上更优，更能说明模型是否真正学到了尾部类别。LDAM scale 消融显示 `s>=5` 会快速退化，而 `s=1/2` 能正常学习，这提示改进策略必须结合训练曲线和尺度超参诊断，而不能只照搬默认配置。

## 9. 进一步工作

可以继续扩展以下方向：

- MiSLAS：先学习表示，再进行分类器重平衡与校准。
- RIDE：多专家长尾识别，用专家分化提升尾部类泛化。
- PaCo：参数化监督对比学习，同时优化表示紧凑性和类别平衡。
- Mixup/CutMix + class-balanced sampling：从数据增强层面提升尾部鲁棒性。
- Decoupled classifier retraining：冻结 backbone，仅用平衡采样重训分类器。
- 使用 ViT 或 CLIP 预训练特征：探索视觉语言预训练对小样本尾部类别的迁移效果。

## 10. 参考文献

[1] Cao K., Wei C., Gaidon A., Arechiga N., Ma T. Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. NeurIPS, 2019. https://arxiv.org/abs/1906.07413

[2] Cui Y., Jia M., Lin T.-Y., Song Y., Belongie S. Class-Balanced Loss Based on Effective Number of Samples. CVPR, 2019. https://arxiv.org/abs/1901.05555

[3] Lin T.-Y., Goyal P., Girshick R., He K., Dollar P. Focal Loss for Dense Object Detection. ICCV, 2017. https://arxiv.org/abs/1708.02002

[4] Menon A. K., Jayasumana S., Rawat A. S., Jain H., Veit A., Kumar S. Long-tail Learning via Logit Adjustment. ICLR, 2021. https://openreview.net/forum?id=37nvvqkCo5

[5] Ren J., Yu C., Sheng S., Ma X., Zhao H., Yi S., Li H. Balanced Meta-Softmax for Long-Tailed Visual Recognition. NeurIPS, 2020. https://arxiv.org/abs/2007.10740

[6] Kang B., Xie S., Rohrbach M., Yan Z., Gordo A., Feng J., Kalantidis Y. Decoupling Representation and Classifier for Long-Tailed Recognition. ICLR, 2020. https://openreview.net/forum?id=r1gRTCVFvB

[7] Wang X., Lian L., Miao Z., Liu Z., Yu S. X. Long-tailed Recognition by Routing Diverse Distribution-Aware Experts. ICLR, 2021. https://openreview.net/forum?id=D9I3drBz4UC

[8] Cui J., Liu S., Tian Z., Zhong Z., Jia J. ResLT: Residual Learning for Long-Tailed Recognition. IEEE TPAMI, 2023. https://arxiv.org/abs/2101.10633
