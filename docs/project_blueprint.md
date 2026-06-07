# 顶会论文级项目大纲

## 核心问题
长尾识别的核心矛盾包括：训练分布与测试目标分布不一致、头部类梯度主导表示学习、尾部类分类间隔不足、模型概率校准偏向头部类。

## 方法路线
1. 基础要求：HOG+PCA+LinearSVM 与 ResNet-32+CE。
2. 主改进：LDAM-DRW，尾部类更大 margin，后期再加权。
3. 前沿增强：Balanced Softmax、Logit Adjustment、Decoupled Classifier、MiSLAS、RIDE、PaCo。

## 实验矩阵
- E1 HOG+SVM: 传统基线。
- E2 ResNet-CE: 神经网络基线。
- E3 ResNet-LDAM-DRW: 主改进。
- E4 ResNet-BalancedSoftmax: 分布修正。
- E5 IR=10/50/100: 难度曲线和鲁棒性分析。

## 评价指标
Top-1、Macro-F1、Balanced Accuracy、Many/Medium/Few Acc、per-class accuracy、confusion matrix、ECE 校准曲线。
