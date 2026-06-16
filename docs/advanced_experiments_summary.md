# 高级补充实验汇总

本文记录四项额外补充实验：Top feature 可读映射、反事实特征翻转、有序分类、Top-2 accuracy 与置信度分析。所有实验均基于当前 full 数据集和已训练模型，没有新增采集。

## 1. Top Feature 可读映射

脚本：

```text
scripts/run_advanced_experiments.py
```

输出：

```text
outputs/analysis/current/metrics/top_features_readable.csv
outputs/analysis/current/advanced/top_features_readable.md
```

目的：

```text
将 answer__q_xxx_Yes 这类机器特征映射回原始问题文本和选项文本，使特征重要性分析可以直接写进报告。
```

示例高重要特征：

| 特征 | 原始问题 | 选项含义 |
|---|---|---|
| `answer__q_6bdc905e3f72_Scary_elements_Learn_more` | Please select all that the game includes: | Scary elements |
| `answer__q_1b293e4babdc_Social_or_Communication` | Category | Social or Communication |
| `answer__q_bd5378991a14_No` | Does the game contain any potentially offensive language? | No |
| `answer__q_507a4755324a_No` | Does the app focus on promoting or selling age-restricted items or activities? | No |

结论：

```text
可解释性分析不再停留在问题 ID 层面，可以明确写出恐怖元素、应用类别、年龄限制活动、攻击性语言等问题与评级预测存在较强关联。
```

## 2. 反事实特征翻转实验

输出：

```text
outputs/analysis/current/metrics/counterfactual_feature_flips.csv
outputs/analysis/current/metrics/counterfactual_summary.json
outputs/analysis/current/figures/counterfactual_prediction_transitions.png
```

方法：

```text
选取 optimized_lightgbm 的 top answer 特征，在 holdout 样本上将单个二元特征从 0 翻到 1 或从 1 翻到 0，观察模型预测是否发生变化。
```

注意：

```text
该实验是模型层面的局部扰动，不保证翻转后的特征组合一定对应真实合法问卷路径。因此它适合解释模型敏感性，而不是证明 Google Play 官方规则。
```

结果：

```text
tested_features = 25
changed_prediction_examples = 62
```

主要预测变化包括：

| 预测变化 | 数量 |
|---|---:|
| `12+ -> 18+` | 24 |
| `16+ -> 18+` | 12 |
| `7+ -> 12+` | 7 |
| `18+ -> 16+` | 4 |
| `12+ -> 3+` | 4 |

结论：

```text
若干高重要问卷特征的单点翻转会导致模型预测从中低年龄段跳到 18+，说明模型确实在这些关键问题上形成了较强的局部决策边界。
```

## 3. 有序分类实验

输出：

```text
outputs/analysis/current/metrics/ordinal_model_metrics.json
outputs/analysis/current/metrics/ordinal_threshold_metrics.csv
outputs/analysis/current/figures/ordinal_vs_multiclass_metrics.png
```

方法：

将年龄分级视为有序等级：

```text
3+ -> 0
7+ -> 1
12+ -> 2
16+ -> 3
18+ -> 4
```

训练四个阈值二分类器：

```text
y >= 7+
y >= 12+
y >= 16+
y >= 18+
```

然后根据通过的阈值数量组合成最终年龄分级。

有序模型 holdout 结果：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.945 |
| Macro-F1 | 0.748 |
| Balanced Accuracy | 0.754 |
| Severe Error Rate | 0.030 |

与当前最佳多分类模型对比：

| 模型 | Accuracy | Macro-F1 | Severe Error |
|---|---:|---:|---:|
| `optimized_lightgbm` | 0.980 | 0.872 | 0.010 |
| `ordinal_lightgbm_thresholds` | 0.945 | 0.748 | 0.030 |

结论：

```text
有序分类思路符合年龄分级的等级结构，但当前实现没有超过直接多分类 LightGBM。报告中可以将其作为探索性实验，说明当前数据和特征下，多分类模型更适合主任务。
```

## 4. Top-2 Accuracy 与置信度分析

输出：

```text
outputs/analysis/current/metrics/top2_confidence_metrics.json
outputs/analysis/current/metrics/top2_predictions.csv
outputs/analysis/current/metrics/confidence_bins.csv
outputs/analysis/current/figures/top2_confidence_bins.png
```

结果：

| 指标 | 数值 |
|---|---:|
| Top-1 Accuracy | 0.980 |
| Top-2 Accuracy | 0.995 |
| Mean Top-1 Confidence | 0.971 |
| Median Top-1 Confidence | 0.996 |

置信度分箱结果：

| Confidence Bin | Count | Avg Confidence | Top-1 Accuracy | Top-2 Accuracy |
|---|---:|---:|---:|---:|
| `(-0.001, 0.5]` | 3 | 0.440 | 0.667 | 1.000 |
| `(0.5, 0.6]` | 2 | 0.502 | 0.500 | 0.500 |
| `(0.7, 0.8]` | 2 | 0.754 | 1.000 | 1.000 |
| `(0.8, 0.9]` | 5 | 0.843 | 1.000 | 1.000 |
| `(0.9, 1.0]` | 187 | 0.990 | 0.989 | 1.000 |

结论：

```text
模型大多数样本置信度很高，且 Top-2 accuracy 达到 0.995。对于少数不确定样本，Top-2 结果通常仍能覆盖真实标签，说明模型在边界样本上具备一定不确定性表达能力。
```

## 5. 可写进报告的新增结论

四项高级实验可以支持以下额外论点：

1. 特征重要性可以映射回原始问题文本，增强可解释性。
2. 关键问卷特征的局部翻转会导致模型预测等级跳变，说明模型确实学习到了局部边界。
3. 有序分类虽然符合标签结构，但当前效果不如直接多分类。
4. Top-2 accuracy 很高，说明模型在边界样本上通常能把真实标签放入前两个候选。
5. 当前最佳模型不仅预测准确，而且具备较强置信度和可解释分析空间。

