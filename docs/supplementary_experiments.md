# 补充实验结果

本文记录当前 full 数据集上的补充实验，包括特征消融、优化模型稳定性和错误案例分析。所有实验均复用 `real_20260615_full` 数据集，没有新增采集。

## 1. 特征消融实验

消融实验使用当前优化后的 LightGBM 参数，并保持相同 holdout split。目标是分析不同特征组对预测性能的贡献。

结果文件：

```text
outputs/analysis/current/metrics/feature_ablation_summary.csv
outputs/analysis/current/metrics/feature_ablation_metrics.json
```

| 特征集合 | 特征数 | Accuracy | Macro-F1 | Balanced Acc | Severe Error |
|---|---:|---:|---:|---:|---:|
| `full` | 1010 | 0.980 | 0.872 | 0.833 | 0.010 |
| `no_strategy` | 1009 | 0.980 | 0.872 | 0.833 | 0.010 |
| `no_aggregate_scores` | 999 | 0.980 | 0.872 | 0.833 | 0.010 |
| `no_descriptor_interactive` | 1008 | 0.965 | 0.787 | 0.781 | 0.020 |
| `answer_only` | 994 | 0.960 | 0.774 | 0.811 | 0.025 |
| `counts_and_scores_only` | 15 | 0.693 | 0.417 | 0.500 | 0.191 |

结论：

1. 完整特征表现最好，`macro_f1 = 0.872`。
2. 去掉 `strategy` 后性能不变，说明模型主要依赖问卷答案和结果相关统计，而不是采样策略。
3. 去掉当前聚合风险分数后性能不变，说明这些聚合特征在当前编码下贡献有限。
4. 去掉 `content_descriptor_count` 和 `interactive_element_count` 后 macro-F1 从 `0.872` 降到 `0.787`，说明结果页派生统计对预测有明显帮助。
5. 只用问卷答案 one-hot 仍有 `macro_f1 = 0.774`，说明问卷答案本身已经包含主要评级信号。
6. 只用计数和聚合分数效果明显不足，说明宏观统计不能替代具体问卷答案。

报告写法建议：

```text
问卷答案 one-hot 是主信息源；结果页内容描述符和互动元素数量提供额外增益；采样策略特征没有带来可见收益，说明模型并非依赖采集批次或策略泄漏。
```

## 2. CV 稳定性实验

优化脚本对每个候选模型执行 5-fold stratified CV，并在 holdout test 上评估最佳参数。

结果文件：

```text
outputs/analysis/current/metrics/optimized_cv_results.csv
outputs/analysis/current/metrics/optimized_cv_stability_summary.csv
```

| 模型 | CV Macro-F1 Mean | CV Macro-F1 Std | Holdout Macro-F1 | Holdout Severe Error |
|---|---:|---:|---:|---:|
| `lightgbm` | 0.741 | 0.076 | 0.872 | 0.010 |
| `xgboost` | 0.704 | 0.066 | 0.776 | 0.025 |
| `extra_trees` | 0.697 | 0.032 | 0.745 | 0.025 |
| `logistic_regression` | 0.698 | 0.051 | 0.745 | 0.020 |
| `decision_tree` | 0.716 | 0.069 | 0.735 | 0.035 |
| `random_forest` | 0.699 | 0.026 | 0.718 | 0.030 |

结论：

1. LightGBM 在 holdout 上最好，但 CV 均值低于 holdout，说明单次 holdout 可能偏乐观。
2. 各模型 CV macro-F1 标准差不小，主要原因是 `3+ / 7+ / 16+` 样本少。
3. Random Forest 和 Extra Trees 的 CV 标准差较小，说明它们更稳定，但 holdout 上不如 LightGBM。
4. 报告中应同时呈现 CV 均值和 holdout 指标，避免只用一次划分做过强结论。

报告写法建议：

```text
由于少数类样本稀缺，单次 holdout 对 macro-F1 影响较大。5-fold CV 结果显示 LightGBM 仍是强模型，但 holdout 结果可能偏乐观，因此最终结论应结合 CV 稳定性讨论。
```

## 3. 错误案例分析

当前最佳优化模型为 `optimized_lightgbm`。holdout test 共 199 条，错分 4 条。

结果文件：

```text
outputs/analysis/current/metrics/optimized_holdout_errors.csv
outputs/analysis/current/metrics/optimized_error_analysis_summary.json
outputs/analysis/current/metrics/optimized_error_transitions.csv
outputs/analysis/current/metrics/optimized_16plus_errors.csv
```

错误转移如下：

| 真实标签 | 预测标签 | 数量 |
|---|---:|---:|
| `16+` | `18+` | 2 |
| `3+` | `16+` | 1 |
| `3+` | `18+` | 1 |

结论：

1. `16+` 的两个错误都被预测为 `18+`，属于保守高估，不是危险低估。
2. 两条严重错误来自 `3+` 被高估到 `16+ / 18+`。
3. 当前最佳模型没有出现 `18+` 被低估到低龄类别的严重错误。
4. 对年龄分级任务而言，保守高估比高风险内容被低估更安全，但仍会影响用户体验。

报告写法建议：

```text
错误案例主要体现为保守高估，而非高风险内容低估。模型对 18+ 的召回非常强，但对少数类边界，尤其 3+ 与 16+，仍存在不稳定性。
```

## 4. 可写进报告的补充实验结论

补充实验可以支持以下论点：

1. 数据中最核心的信息来自具体问卷答案，而不是采样策略。
2. 内容描述符和互动元素数量对预测有额外帮助。
3. 简单聚合风险分数在当前版本中贡献有限，后续可以考虑基于问题文本重建更强的语义聚合特征。
4. LightGBM 在 holdout 上最强，但 CV 稳定性提醒我们需要谨慎解释少数类结果。
5. 错误样本主要是保守高估，未观察到严重的高风险低估案例。

