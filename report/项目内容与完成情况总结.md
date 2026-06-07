# CIFAR-100-LT 图像长尾分类项目内容与完成情况总结

## 1. 项目主题

本项目对应结课考核题目“图像长尾分类实验”，围绕 CIFAR-100-LT 数据集构建、传统机器学习方法、神经网络方法和长尾分类改进策略展开实验。

项目核心目标是：在类别样本数严重不均衡的训练集上训练分类模型，并分别在平衡测试集和同分布不平衡测试集上评估模型表现，分析不同方法对头部类、中部类和尾部类的影响。

## 2. 题目基本要求与完成情况

| 题目要求 | 当前完成情况 | 说明 |
|---|---|---|
| 使用 exp 或 step 构建不平衡 CIFAR-100-LT 数据集 | 已完成 | 项目同时支持 `exp` 和 `step` 两种构建方式，正式实验采用 `exp IR=100` |
| 选择 1 种传统机器学习方法 | 已完成 | 使用 `HOG + PCA + LinearSVM` |
| 选择 1 种神经网络方法 | 已完成 | 使用 CIFAR 风格 `ResNet-32 + Cross Entropy` |
| 报告训练集表现 | 已完成 | 每个方法均生成 train split 指标 |
| 报告平衡测试集表现 | 已完成 | 使用 CIFAR-100 原始测试集，每类 100 张 |
| 报告不平衡测试集表现 | 已完成 | 按训练集同分布抽样构建 |
| 比较两种方法表现 | 已完成 | 比较传统方法、CE 神经网络基线和多种长尾方法 |
| 针对一种方法采用至少一种策略提升性能 | 已完成且超额 | 实现并评估 LDAM-DRW、Balanced Softmax、Logit Adjustment、Class-Balanced Focal |
| 分析改进是否有效 | 已完成 | 分析了 CE、LDAM-DRW、Balanced Softmax、CB-Focal 的优缺点 |
| 说明数据预处理、模型设置、评价指标和实验结论 | 已完成 | 已写入 Markdown 报告和 Word 报告 |
| 使用柱状图、折线图等展示数据分布和结果 | 已完成且超额 | 包括分布图、指标对比图、逐类准确率图、混淆矩阵、reliability diagram、LDAM scale 消融图 |

结论：项目已经完整满足题目基础要求，并在方法数量、指标体系、可视化、消融实验和工程交付方面明显超额完成。

## 3. 本组完成的主要工作内容

### 3.1 数据集构建

本组基于 CIFAR-100 构建 CIFAR-100-LT 长尾数据集。

原始 CIFAR-100 包含：

- 100 个类别；
- 每类 500 张训练图像；
- 每类 100 张测试图像。

正式实验采用 `exp IR=100` 设置，类别样本数按指数形式递减：

```text
n_i = n_max * (1 / IR) ** (i / (C - 1))
```

其中：

- `C = 100`
- `n_max = 500`
- `IR = 100`

因此，头部类约有 500 张训练图像，尾部类约有 5 张训练图像。

项目同时支持：

- `exp` 长尾构造；
- `step` 长尾构造。

测试集分为两种：

- 平衡测试集：使用原始 CIFAR-100 测试集，每类 100 张；
- 不平衡测试集：按照训练集同样的长尾分布从测试集中抽样。

### 3.2 传统机器学习方法

传统方法采用：

```text
HOG + PCA + LinearSVM
```

处理流程：

1. 对 32x32 RGB 图像提取 HOG 特征；
2. 使用 `StandardScaler` 标准化；
3. 使用 PCA 降维；
4. 使用带 class-balanced 权重的 Linear SVM 分类。

该方法作为传统机器学习基线，满足课程对“传统方法”的要求。

### 3.3 神经网络基线

神经网络方法采用：

```text
ResNet-32 + Cross Entropy
```

训练设置：

- 模型：CIFAR 风格 ResNet-32；
- 优化器：SGD；
- 初始学习率：0.1；
- Momentum：0.9；
- Weight decay：0.0002；
- 训练轮数：200 epoch；
- 学习率里程碑：160 和 180 epoch；
- 数据增强：随机裁剪、水平翻转、标准化。

### 3.4 长尾改进策略

本组不只实现一种改进策略，而是实现并评估了多种长尾分类方法。

已完成方法包括：

- LDAM-DRW；
- 调参版 LDAM-DRW(s=1)；
- Balanced Softmax；
- Logit Adjustment；
- Class-Balanced Focal；
- LDAM scale 消融实验。

其中，Balanced Softmax / Logit Adjustment τ=1 是当前实验中综合表现最好的长尾改进方法。

## 4. 评价指标

项目没有只报告 overall accuracy，而是使用了更适合长尾分类的多指标体系。

主要指标包括：

- Top-1 Accuracy；
- Top-5 Accuracy；
- Macro-F1；
- Weighted-F1；
- Balanced Accuracy；
- Many Acc；
- Medium Acc；
- Few Acc；
- ECE；
- Per-class Accuracy；
- Confusion Matrix；
- Reliability Diagram。

类别分组方式：

| 分组 | 训练样本数 |
|---|---:|
| Many | `>= 100` |
| Medium | `20 到 99` |
| Few | `< 20` |

这些指标能够更准确地反映长尾分类中尾部类别的学习情况。

## 5. 正式实验结果

### 5.1 主实验结果

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

### 5.2 LDAM scale 消融实验

为解释原始 LDAM-DRW `s=30` 未收敛的问题，本组补充进行了 30 epoch 的 scale 消融实验。

| LDAM scale | Balanced Top-1 | Macro-F1 | Few Acc | ECE |
|---:|---:|---:|---:|---:|
| 0.5 | 34.50 | 29.31 | 9.20 | 36.63 |
| 1 | 35.90 | 31.34 | 10.73 | 12.97 |
| 2 | 34.94 | 30.69 | 12.40 | 12.12 |
| 5 | 4.44 | 1.45 | 0.03 | 2.91 |
| 10 | 1.00 | 0.02 | 0.00 | 0.42 |
| 30 | 1.01 | 0.04 | 0.00 | 21.23 |

结论：

- `s=0.5/1/2` 能够正常学习；
- `s>=5` 后性能断崖式下降；
- `s=10` 和 `s=30` 接近随机或单类预测；
- 原始 `s=30` 失败主要来自当前模型、学习率和 loss scale 组合下的优化困难。

## 6. 实验分析总结

### 6.1 传统方法与神经网络方法对比

传统 `HOG+PCA+LinearSVM` 在平衡测试集上的 Top-1 仅为 12.63%，明显弱于 ResNet-32+CE 的 40.66%。

这说明：

- 手工特征难以捕捉 CIFAR-100 的高层语义；
- 深度神经网络在图像分类任务中具有明显优势；
- 传统方法可以作为课程要求中的基线，但不是解决长尾分类的主要方向。

### 6.2 CE 基线的长尾偏置

ResNet-32+CE 在不平衡测试集上的 Top-1 为 64.71%，看似表现很好。

但在平衡测试集上：

- Top-1 为 40.66%；
- Few Acc 仅为 9.63%。

这说明 CE 的高 overall accuracy 很大程度上来自头部类别，不能说明模型真正解决了尾部类别问题。

### 6.3 LDAM-DRW 的诊断价值

原始 LDAM-DRW `s=30` 没有收敛，平衡测试集 Top-1 仅为 1.00%。

但调参版 `s=1` 达到：

- Balanced Top-1：44.60%；
- Macro-F1：42.28%；
- Few Acc：19.87%。

这说明 LDAM 的 margin 思路本身并非无效，问题主要在于 loss scale 与当前训练设置不匹配。

### 6.4 Balanced Softmax / Logit Adjustment 的优势

Balanced Softmax 和 Logit Adjustment τ=1 的结果完全一致：

- Balanced Top-1：44.89%；
- Macro-F1：43.93%；
- Few Acc：27.27%；
- ECE：22.41%。

这是本项目中综合表现最好的方法。

这说明显式引入类别先验、修正训练分布偏移，对于 CIFAR-100-LT exp IR=100 是有效的。

### 6.5 Class-Balanced Focal 的负结果

Class-Balanced Focal 的 Few Acc 为 19.13%，高于 CE 的 9.63%。

但其平衡测试 Top-1 只有 21.55%，明显低于 CE 和 Balanced Softmax。

这说明：

- 强重加权确实能照顾尾部类别；
- 但过强的类别权重和 focal 调制会严重损害整体判别能力；
- 长尾分类不能只追求尾部提升，还要平衡整体准确率和类别公平性。

## 7. 已完成的可视化内容

项目已生成以下可视化：

| 图表 | 文件路径 | 说明 |
|---|---|---|
| exp 类别分布图 | `report/figures/generated/distribution_exp_ir100.png` | 展示 exp IR=100 长尾分布 |
| step 类别分布图 | `report/figures/generated/distribution_step_ir100.png` | 展示 step IR=100 长尾分布 |
| Top-1 对比图 | `report/figures/generated/comparison_top1.png` | 比较不同方法 Top-1 |
| Macro-F1 对比图 | `report/figures/generated/comparison_macro_f1.png` | 比较宏平均 F1 |
| Few Acc 对比图 | `report/figures/generated/comparison_few_acc.png` | 比较尾部类别表现 |
| CE 逐类准确率图 | `report/figures/generated/per_class_ce_resnet32_ir100_balanced_test.png` | 展示 CE 对不同类别的准确率 |
| LDAM(s=1) 逐类准确率图 | `report/figures/generated/per_class_ldam_drw_resnet32_ir100_s1_balanced_test.png` | 展示调参版 LDAM 的逐类表现 |
| Balanced Softmax 混淆矩阵 | `report/figures/generated/confusion_balanced_softmax_resnet32_ir100_balanced_test.png` | 分析类别混淆 |
| Balanced Softmax 校准图 | `report/figures/generated/reliability_balanced_softmax_resnet32_ir100_balanced_test.png` | 分析模型置信度校准 |
| LDAM scale 消融图 | `report/figures/generated/ldam_scale_ablation.png` | 展示 scale 对 LDAM 的影响 |

## 8. 相对于基本要求的超额完成内容

本项目相对于课程基础要求，主要超额完成了以下内容：

1. 不只实现一种改进策略，而是实现并评估了多种长尾分类方法；
2. 不只报告 accuracy，而是报告多种长尾分类指标；
3. 不只比较平衡测试和不平衡测试，还分析了二者差异；
4. 补充了 LDAM scale 消融实验；
5. 保留并分析了失败实验，而不是只展示成功结果；
6. 生成了混淆矩阵和 reliability diagram；
7. 形成了完整工程交付，包括代码、配置、结果、图表、报告和 zip 包；
8. 可直接扩展为一篇完整实验论文。

## 9. 当前项目交付文件

主要交付文件包括：

| 文件 | 说明 |
|---|---|
| `README.md` | 项目说明 |
| `configs/` | 实验配置 |
| `src/longtail/` | 核心代码 |
| `scripts/aggregate_results.py` | 结果聚合脚本 |
| `scripts/plot_diagnostics.py` | 混淆矩阵、校准图和消融图生成脚本 |
| `results/summary.csv` | 汇总指标 |
| `results/summary.md` | Markdown 汇总表 |
| `report/report.md` | 中文实验报告 |
| `report/机器学习课程作业_图像长尾分类实验报告.docx` | Word 报告 |
| `report/figures/generated/` | 自动生成图表 |
| `longtail_cifar100_project_final.zip` | 最终压缩包 |

## 10. 项目整体结论

本组项目已经完成课程题目“图像长尾分类实验”的全部基础要求，并在方法数量、评价指标、可视化、消融实验和工程交付方面明显超额完成。

实验表明：

1. 传统 HOG+PCA+LinearSVM 难以处理 CIFAR-100-LT 细粒度长尾分类；
2. ResNet-32+CE 在不平衡测试集上整体准确率较高，但尾部类别表现不足；
3. LDAM-DRW 需要合适的 scale，原始 `s=30` 在本设置下未收敛；
4. LDAM-DRW(s=1) 能够有效改善平衡测试集和尾部类别表现；
5. Balanced Softmax / Logit Adjustment τ=1 是当前实验中综合表现最好的方法；
6. Class-Balanced Focal 提升部分尾部表现，但整体性能下降明显；
7. 长尾分类评估不能只看 overall accuracy，应结合平衡测试集、Macro-F1、Balanced Accuracy、Few Acc、逐类准确率、混淆矩阵和校准指标综合判断。

因此，本项目不仅满足课程实验要求，也具备撰写完整实验论文的基础。

