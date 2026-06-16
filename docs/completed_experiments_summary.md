# 已完成实验与结果汇总

本文汇总当前项目已经完成的实验、数据、指标和主要结论。当前所有实验均基于 full 数据集 `real_20260615_full`，没有使用新的数据采集。

## 0. 核心结论一览

当前实验部分已经完成课程要求，并且增加了若干补充实验用于增强报告说服力。

| 项目 | 当前结果 |
|---|---|
| 有效样本数 | `1322` |
| 主标签 | `IARC Generic / result_age_rating` |
| 主要类别问题 | `18+` 占 `76.6%`，类别明显不均衡 |
| 第一版最佳 macro-F1 | `xgboost = 0.833` |
| 优化后最佳模型 | `optimized_lightgbm` |
| 优化后最佳 accuracy | `0.980` |
| 优化后最佳 macro-F1 | `0.872` |
| 优化后 severe error rate | `0.010` |
| 最难类别 | `16+`，测试集 support 只有 `4` |
| 最重要补充实验 | 特征消融、CV 稳定性、错误案例分析 |

可以在报告中使用的核心表述：

```text
基于 1322 条真实问卷样本，实验训练了 Google Play / IARC 年龄分级问卷的黑盒替代模型。优化后的 LightGBM 在 holdout test 上达到 0.980 accuracy 和 0.872 macro-F1。特征消融表明，问卷答案是主要信号源，内容描述符和互动元素数量提供额外增益。错误分析显示模型主要倾向保守高估，而没有观察到 18+ 被严重低估为低龄类别的情况。
```

## 1. 当前数据集

当前主数据文件：

```text
data/raw/real_20260615_full.samples.jsonl
data/processed/real_20260615_full.dataset.csv
data/processed/real_20260615_full.features.csv
```

数据验证结果：

```text
valid_samples = 1322
invalid_samples = 0
missing_label_count = 0
duplicate_answer_count = 0
unique_answer_count = 1322
```

主标签使用 `IARC Generic` 口径，对应字段为：

```text
result_age_rating
```

标签分布：

| 年龄分级 | 样本数 | 占比 |
|---|---:|---:|
| `3+` | 37 | 2.8% |
| `7+` | 19 | 1.4% |
| `12+` | 225 | 17.0% |
| `16+` | 28 | 2.1% |
| `18+` | 1013 | 76.6% |

当前建模特征矩阵：

```text
X shape = 1322 x 1010
features.csv shape = 1322 x 1011
```

其中 `features.csv` 多出的 1 列是标签。

当前训练/测试划分：

```text
random_seed = 42
test_size = 0.15
train size = 1123
test size = 199
```

测试集标签 support：

| 年龄分级 | 测试集样本数 |
|---|---:|
| `3+` | 6 |
| `7+` | 3 |
| `12+` | 34 |
| `16+` | 4 |
| `18+` | 152 |

说明：`16+` 测试集只有 4 条，因此该类别的 precision、recall 和 F1 容易波动。

## 1.1 评估指标定义

由于标签分布极不均衡，本文不把 accuracy 作为唯一指标，而是同时报告 macro-F1、balanced accuracy、weighted-F1、年龄等级误差和严重错误率。

Accuracy：

```text
Accuracy = (1 / n) * sum_i 1[y_i = y_hat_i]
```

每个类别的 precision、recall 和 F1：

```text
Precision_k = TP_k / (TP_k + FP_k)
Recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 * Precision_k * Recall_k / (Precision_k + Recall_k)
```

Macro-F1：

```text
Macro-F1 = (1 / K) * sum_k F1_k
```

Balanced accuracy：

```text
Balanced Accuracy = (1 / K) * sum_k Recall_k
```

年龄等级映射：

```text
3+  -> 0
7+  -> 1
12+ -> 2
16+ -> 3
18+ -> 4
```

平均年龄等级误差：

```text
MAE = (1 / n) * sum_i |order(y_i) - order(y_hat_i)|
```

严重错误率：

```text
Severe Error Rate = (1 / n) * sum_i 1[ |order(y_i) - order(y_hat_i)| >= 2 ]
```

报告写作建议：

```text
Accuracy 体现总体预测正确率，但在 18+ 占多数的数据集中容易偏乐观。Macro-F1 和 balanced accuracy 更能反映少数类表现。Severe error rate 用于衡量年龄分级任务中更严重的跨等级错误。
```

## 2. 数据处理与特征工程

每条样本保留完整问卷答案 `answers_json`。建模时将问卷答案展开为 one-hot 特征：

```text
answer__question_id_option = 0/1
```

另外保留以下统计特征：

```text
visible_question_count
skipped_question_count
content_descriptor_count
interactive_element_count
violence_score
sexual_content_score
language_score
drug_score
gambling_score
fear_score
ugc_score
interaction_score
high_risk_count
medium_risk_count
triggered_branch_count
```

已经修复的关键数据读取问题：

```python
pd.read_csv(path, keep_default_na=False, na_values=[""])
```

原因是 pandas 默认会把字符串 `"None"` 当作缺失值，但问卷中 `"None"` 可能是合法选项。当前设置保证：

```text
空单元格 -> 缺失值
字符串 "None" -> 合法类别
```

同时，特征名已统一清洗为 LightGBM / XGBoost 可接受的格式，避免特殊字符导致训练失败。

## 3. 第一版模型对比实验

实验脚本：

```text
scripts/train_models.py
```

指标文件：

```text
outputs/analysis/current/metrics/model_metrics.json
```

模型列表：

```text
majority
stratified
logistic_regression
decision_tree
random_forest
extra_trees
xgboost
lightgbm
```

第一版 holdout test 结果：

| 模型 | Accuracy | Macro-F1 | Balanced Acc | Weighted-F1 | MAE | Severe Error |
|---|---:|---:|---:|---:|---:|---:|
| `xgboost` | 0.970 | 0.833 | 0.777 | 0.965 | 0.065 | 0.015 |
| `lightgbm` | 0.975 | 0.824 | 0.783 | 0.971 | 0.050 | 0.010 |
| `extra_trees` | 0.940 | 0.812 | 0.760 | 0.935 | 0.126 | 0.045 |
| `logistic_regression` | 0.970 | 0.748 | 0.765 | 0.960 | 0.070 | 0.015 |
| `random_forest` | 0.910 | 0.642 | 0.667 | 0.900 | 0.171 | 0.060 |
| `decision_tree` | 0.839 | 0.608 | 0.667 | 0.877 | 0.296 | 0.075 |
| `stratified` | 0.633 | 0.193 | 0.193 | 0.626 | 0.784 | 0.327 |
| `majority` | 0.764 | 0.173 | 0.200 | 0.662 | 0.528 | 0.216 |

第一版结论：

1. `xgboost` 的 macro-F1 最高，达到 `0.833`。
2. `lightgbm` 的 accuracy 最高，达到 `0.975`，severe error 最低，达到 `0.010`。
3. 所有正式模型均明显优于 `majority` 和 `stratified` baseline。
4. 由于数据集类别不均衡，报告中应优先讨论 macro-F1、balanced accuracy 和 severe error，而不是只讨论 accuracy。

## 4. 模型优化实验

实验脚本：

```text
scripts/optimize_models.py
```

输出文件：

```text
outputs/analysis/current/metrics/optimized_model_metrics.json
outputs/analysis/current/metrics/optimized_cv_results.csv
outputs/analysis/current/metrics/optimized_holdout_errors.csv
outputs/analysis/current/models/optimized_*.joblib
```

优化方法：

1. 使用 `5-fold StratifiedKFold` 在训练集内部做参数选择。
2. 使用 `macro_f1` 作为主要优化目标。
3. 对部分模型比较 `class_weight`、`sample_weight` 和无权重设置。
4. 对最佳参数模型重新训练，并在 holdout test 上评估。

优化后 holdout test 结果：

| 模型 | Accuracy | Macro-F1 | Balanced Acc | Severe Error | 最优参数摘要 |
|---|---:|---:|---:|---:|---|
| `optimized_lightgbm` | 0.980 | 0.872 | 0.833 | 0.010 | `max_depth=5, num_leaves=15` |
| `optimized_xgboost` | 0.955 | 0.776 | 0.778 | 0.025 | `max_depth=4, colsample_bytree=1.0` |
| `optimized_extra_trees` | 0.955 | 0.745 | 0.716 | 0.025 | `max_depth=None, min_samples_leaf=1` |
| `optimized_logistic_regression` | 0.965 | 0.745 | 0.764 | 0.020 | `C=3.0` |
| `optimized_decision_tree` | 0.925 | 0.735 | 0.708 | 0.035 | `max_depth=12, min_samples_leaf=3` |
| `optimized_random_forest` | 0.950 | 0.718 | 0.682 | 0.030 | `max_depth=None, min_samples_leaf=1` |

优化前后 macro-F1 对比：

| 模型 | 原 Macro-F1 | 优化 Macro-F1 | 变化 |
|---|---:|---:|---:|
| `decision_tree` | 0.608 | 0.735 | +0.128 |
| `random_forest` | 0.642 | 0.718 | +0.076 |
| `lightgbm` | 0.824 | 0.872 | +0.048 |
| `logistic_regression` | 0.748 | 0.745 | -0.003 |
| `xgboost` | 0.833 | 0.776 | -0.057 |
| `extra_trees` | 0.812 | 0.745 | -0.067 |

优化实验结论：

1. `optimized_lightgbm` 是当前整体最佳模型，macro-F1 达到 `0.872`。
2. `decision_tree` 优化收益最大，macro-F1 从 `0.608` 提升到 `0.735`，适合作为可解释规则模型。
3. `random_forest` 也有明显提升，从 `0.642` 提升到 `0.718`。
4. `xgboost` 和 `extra_trees` 在本轮加权/调参后 holdout macro-F1 下降，说明调参不一定总是提升，需要结合验证集和 holdout 共同判断。

## 5. 当前最佳模型逐类别结果

当前最佳模型为：

```text
optimized_lightgbm
```

逐类别结果：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `3+` | 1.000 | 0.667 | 0.800 | 6 |
| `7+` | 1.000 | 1.000 | 1.000 | 3 |
| `12+` | 1.000 | 1.000 | 1.000 | 34 |
| `16+` | 0.667 | 0.500 | 0.571 | 4 |
| `18+` | 0.981 | 1.000 | 0.990 | 152 |

结论：

1. `18+` 识别非常稳定，recall 达到 `1.000`。
2. `12+` 在 holdout 上完全识别正确。
3. `16+` 仍然是最难类别，recall 为 `0.500`，但相比优化前的 `0.250` 有提升。
4. `3+` support 只有 6，recall 为 `0.667`，仍有波动风险。

## 6. 特征消融实验

实验脚本：

```text
scripts/run_feature_ablation.py
```

输出文件：

```text
outputs/analysis/current/metrics/feature_ablation_summary.csv
outputs/analysis/current/metrics/feature_ablation_metrics.json
```

特征消融结果：

| 特征集合 | 特征数 | Accuracy | Macro-F1 | Balanced Acc | Severe Error |
|---|---:|---:|---:|---:|---:|
| `full` | 1010 | 0.980 | 0.872 | 0.833 | 0.010 |
| `no_strategy` | 1009 | 0.980 | 0.872 | 0.833 | 0.010 |
| `no_aggregate_scores` | 999 | 0.980 | 0.872 | 0.833 | 0.010 |
| `no_descriptor_interactive` | 1008 | 0.965 | 0.787 | 0.781 | 0.020 |
| `answer_only` | 994 | 0.960 | 0.774 | 0.811 | 0.025 |
| `counts_and_scores_only` | 15 | 0.693 | 0.417 | 0.500 | 0.191 |

特征消融结论：

1. 完整特征表现最好。
2. 去掉 `strategy` 后性能不变，说明模型没有依赖采样策略泄漏。
3. 去掉当前聚合风险分数后性能不变，说明这些聚合分数在当前版本中贡献有限。
4. 去掉 `content_descriptor_count` 和 `interactive_element_count` 后 macro-F1 从 `0.872` 降到 `0.787`，说明结果页派生统计有明显贡献。
5. 只使用问卷答案 one-hot 仍能达到 `macro_f1 = 0.774`，说明问卷答案本身包含主要评级信号。
6. 只使用计数和聚合分数效果较差，macro-F1 只有 `0.417`，说明宏观统计不能替代具体问卷答案。

## 7. CV 稳定性实验

汇总脚本：

```text
scripts/summarize_experiment_results.py
```

输出文件：

```text
outputs/analysis/current/metrics/optimized_cv_stability_summary.csv
```

CV 稳定性结果：

| 模型 | CV Macro-F1 Mean | CV Macro-F1 Std | Holdout Macro-F1 | Holdout Severe Error |
|---|---:|---:|---:|---:|
| `lightgbm` | 0.741 | 0.076 | 0.872 | 0.010 |
| `xgboost` | 0.704 | 0.066 | 0.776 | 0.025 |
| `extra_trees` | 0.697 | 0.032 | 0.745 | 0.025 |
| `logistic_regression` | 0.698 | 0.051 | 0.745 | 0.020 |
| `decision_tree` | 0.716 | 0.069 | 0.735 | 0.035 |
| `random_forest` | 0.699 | 0.026 | 0.718 | 0.030 |

CV 稳定性结论：

1. LightGBM 在 holdout 上最好，但 CV macro-F1 均值低于 holdout，说明单次 holdout 可能偏乐观。
2. 少数类样本稀缺导致 CV 波动较大，尤其影响 macro-F1。
3. Random Forest 和 Extra Trees 的 CV 标准差较小，但最终 holdout 性能不如 LightGBM。
4. 报告中应同时呈现 CV 均值和 holdout 指标，避免只用一次划分得出过强结论。

## 8. 错误案例分析

输出文件：

```text
outputs/analysis/current/metrics/optimized_holdout_errors.csv
outputs/analysis/current/metrics/optimized_error_analysis_summary.json
outputs/analysis/current/metrics/optimized_error_transitions.csv
outputs/analysis/current/metrics/optimized_16plus_errors.csv
```

当前最佳模型 `optimized_lightgbm` 在 holdout test 上共错分 4 条：

| 真实标签 | 预测标签 | 数量 |
|---|---:|---:|
| `16+` | `18+` | 2 |
| `3+` | `16+` | 1 |
| `3+` | `18+` | 1 |

错误分析结论：

1. `16+` 的两个错误都被预测为 `18+`，属于保守高估。
2. 两条严重错误来自 `3+` 被高估到 `16+ / 18+`。
3. 当前最佳模型没有出现 `18+` 被低估到低龄类别的严重错误。
4. 对年龄分级任务而言，保守高估比高风险内容被低估更安全，但仍会影响用户体验。

## 9. 模型解释实验

解释脚本：

```text
scripts/explain_models.py
```

输出目录：

```text
outputs/analysis/current/explanations/
```

当前解释产物包括：

```text
decision_tree_rules.txt
decision_tree_feature_importance.csv
random_forest_feature_importance.csv
extra_trees_feature_importance.csv
xgboost_feature_importance.csv
lightgbm_feature_importance.csv
logistic_regression_permutation_importance.csv
```

解释方法：

1. 决策树导出可读规则文本，用于近似黑盒规则。
2. 树模型和 boosting 模型输出原生 feature importance。
3. 逻辑回归使用 permutation importance。
4. 为避免解释阶段超时，permutation importance 限制为 `MAX_PERMUTATION_SAMPLES = 200` 和 `PERMUTATION_REPEATS = 1`。

## 10. 图表生成

基础图表脚本：

```text
scripts/make_figures.py
```

综合实验图表脚本：

```text
scripts/make_experiment_figures.py
```

输出目录：

```text
outputs/analysis/current/figures/
```

当前基础图表包括：

```text
label_distribution.png
model_performance.png
confusion_matrix_xgboost.png
decision_tree_feature_importance.png
extra_trees_feature_importance.png
lightgbm_feature_importance.png
random_forest_feature_importance.png
xgboost_feature_importance.png
```

当前综合实验图表包括：

```text
experiment_label_distribution.png
base_vs_optimized_macro_f1.png
optimized_model_performance.png
optimized_severe_error_rate.png
feature_ablation_macro_f1.png
cv_stability_macro_f1.png
optimized_error_transitions.png
google_play_vs_iarc_match.png
authority_presence_counts.png
region_rating_prediction_performance.png
```

推荐放入报告正文的图表：

| 图表 | 建议放置位置 | 用途 |
|---|---|---|
| `experiment_label_distribution.png` | 数据集分析 | 展示类别不均衡 |
| `base_vs_optimized_macro_f1.png` | 模型训练与优化 | 展示优化前后提升 |
| `optimized_model_performance.png` | 模型训练与评估 | 展示优化模型综合表现 |
| `feature_ablation_macro_f1.png` | 特征工程分析 | 证明不同特征组贡献 |
| `cv_stability_macro_f1.png` | 评估稳定性 | 展示 CV 与 holdout 的差异 |
| `optimized_error_transitions.png` | 错误案例分析 | 展示最佳模型错分方向 |
| `decision_tree_feature_importance.png` | 黑盒机制解释 | 展示可解释规则模型的重要特征 |
| `lightgbm_feature_importance.png` | 黑盒机制解释 | 展示最佳模型族的重要特征 |
| `google_play_vs_iarc_match.png` | 扩展分析 | 展示 Google Play 与 IARC 口径差异 |
| `region_rating_prediction_performance.png` | 扩展分析 | 展示不同地区评级机构的预测效果 |

图表解释注意：

1. `confusion_matrix_xgboost.png` 对应第一版最佳 macro-F1 模型 `xgboost`，展示完整预测矩阵。
2. `optimized_error_transitions.png` 对应优化后最佳模型 `optimized_lightgbm`，只展示错分样本，不展示预测正确样本。
3. `model_performance.png` 对应第一版 `model_metrics.json`。
4. `optimized_model_performance.png` 对应优化后 `optimized_model_metrics.json`。
5. 特征重要性图展示的是模型使用某些特征降低分类不确定性的程度，不应直接解释为严格因果关系。

## 11. 多机构评级分析

分析脚本：

```text
scripts/analyze_region_ratings.py
```

输出文件：

```text
outputs/analysis/current/metrics/region_rating_summary.json
```

多机构评级统计：

```text
total_success_samples = 1322
samples_with_region_ratings = 1322
missing_region_ratings = 0
distinct_authority_count = 10
```

样本中包含的评级机构包括：

```text
IARC Generic
Google Play
ESRB
PEGI
USK
ClassInd
ACB
DGSC
GRAC
Gmedia
```

Google Play 与 IARC Generic 对比：

```text
both_present_count = 1322
exact_match_count = 748
mismatch_count = 574
```

主要差异来源是 Google Play 中存在 `Rated for 19+`，而当前主标签使用 IARC Generic 的 `18+` 口径。当前实验暂不改变主标签，但该结果可以作为报告中的扩展分析。

## 11.1 地区评级预测任务

实验脚本：

```text
scripts/train_region_rating_models.py
```

详细文档：

```text
docs/region_rating_prediction_summary.md
```

输出文件：

```text
outputs/analysis/current/metrics/region_rating_model_metrics.json
outputs/analysis/current/metrics/region_rating_model_summary.csv
outputs/analysis/current/metrics/region_rating_confusion/
outputs/analysis/current/metrics/region_rating_predictions/
outputs/analysis/current/models/region_rating_*.joblib
outputs/analysis/current/figures/region_rating_prediction_performance.png
```

该任务使用 `result_region_ratings` 中的不同机构标签分别训练模型。为了避免标签泄漏，训练地区评级模型时显式移除了主标签 `result_age_rating`，只使用问卷答案和派生统计特征。

训练设置：

```text
model = LightGBM
random_seed = 42
test_size = 0.15
sample_weight = balanced
min_samples = 100
min_classes = 2
min_class_count = 3
```

第一版地区评级预测结果：

| 机构 | 样本数 | 类别数 | Accuracy | Macro-F1 | Balanced Acc |
|---|---:|---:|---:|---:|---:|
| `IARC Generic` | 1322 | 5 | 0.975 | 0.829 | 0.767 |
| `ESRB` | 1322 | 5 | 0.884 | 0.822 | 0.807 |
| `Google Play` | 1322 | 6 | 0.960 | 0.811 | 0.787 |
| `PEGI` | 1322 | 6 | 0.960 | 0.748 | 0.744 |
| `USK` | 1322 | 5 | 0.970 | 0.738 | 0.749 |
| `ClassInd` | 1322 | 6 | 0.915 | 0.674 | 0.691 |
| `DGSC` | 514 | 5 | 0.897 | 0.585 | 0.676 |
| `ACB` | 514 | 6 | 0.910 | 0.522 | 0.516 |

跳过机构：

| 机构 | 样本数 | 类别数 | 最小类别样本数 | 原因 |
|---|---:|---:|---:|---|
| `GRAC` | 514 | 4 | 2 | `min_class_count < 3` |
| `Gmedia` | 514 | 6 | 1 | `min_class_count < 3` |

地区评级预测结论：

1. 当前问卷特征不仅能预测主标签，也能预测多个地区/机构评级标签。
2. `ESRB` 的 macro-F1 达到 `0.822`，说明它在各类别之间表现相对均衡。
3. `Google Play` 的 macro-F1 达到 `0.811`，且它包含 `Rated for 19+`，不是 IARC Generic 标签的简单复制。
4. `ACB` 和 `DGSC` 的 accuracy 不低，但 macro-F1 较低，主要受 514 样本规模和极端类别不均衡影响。
5. `GRAC` 和 `Gmedia` 因最小类别样本数过低，本轮只保留统计分析，不报告模型指标。

## 12. 已通过的验证

当前已运行：

```text
python -m pytest -q
```

结果：

```text
4 passed
```

同时已经跑通：

```text
validate_dataset.py
build_dataset.py
train_models.py
optimize_models.py
explain_models.py
make_figures.py
run_feature_ablation.py
summarize_experiment_results.py
analyze_region_ratings.py
train_region_rating_models.py
```

## 13. 局限性与注意事项

当前实验已经足够完整，但报告中需要诚实说明以下局限性。

### 13.1 类别不均衡

`18+` 样本占 `76.6%`，而 `3+ / 7+ / 16+` 样本较少。尤其 holdout test 中：

```text
3+ support = 6
7+ support = 3
16+ support = 4
```

因此少数类指标存在波动，不能过度解读单次 holdout 上的少数类 F1。

报告建议表述：

```text
虽然优化后的模型在 holdout test 上表现较好，但由于少数类样本数量有限，3+、7+、16+ 的逐类别指标仍存在较大不确定性。本文通过 macro-F1、balanced accuracy 和 5-fold CV 缓解单一 accuracy 指标的偏差。
```

### 13.2 黑盒近似不等于真实规则

模型学到的是问卷答案与评级结果之间的统计映射，不等于 Google Play / IARC 的真实内部规则。

报告建议表述：

```text
本文训练的是黑盒替代模型，而不是官方规则复现。特征重要性和决策树规则只能作为机制线索，不能解释为官方评级规则。
```

### 13.3 特征重要性不是因果关系

例如 `Scary elements`、`visible_question_count` 和 `Social or Communication` 在树模型中重要，说明它们对模型切分样本有帮助，但不代表单独改变该特征一定会导致评级变化。

更严谨的因果分析需要反事实样本实验。当前项目已经有足够实验支撑报告，不建议继续扩张实验范围。

### 13.4 主标签口径选择

当前主标签使用 `IARC Generic`。样本中也有 Google Play、ESRB、PEGI、USK 等多机构评级。Google Play 与 IARC Generic 有 `574` 条显示不一致，主要来自 Google Play 的 `19+` 口径。

报告建议表述：

```text
本文主任务固定使用 IARC Generic 年龄分级，以保证标签空间为 3+ / 7+ / 12+ / 16+ / 18+。多机构评级差异作为扩展分析，不参与主模型训练。
```

## 14. 报告写作建议结构

可以按以下结构组织最终实验报告：

```text
1. Introduction
2. Task Definition and Black-box Setting
3. Data Collection and Dataset Construction
4. Feature Engineering
5. Model Training and Evaluation Metrics
6. Baseline Model Comparison
7. Model Optimization
8. Feature Ablation Study
9. Error Analysis and CV Stability
10. Interpreting the Black Box
11. Regional Rating Differences
12. Limitations
13. Conclusion
```

最推荐放入正文的结果表：

1. 标签分布表。
2. 第一版模型对比表。
3. 优化模型对比表。
4. 当前最佳模型逐类别结果表。
5. 特征消融表。
6. CV 稳定性表。
7. 错误转移表。

最推荐放入正文的图：

1. `experiment_label_distribution.png`
2. `base_vs_optimized_macro_f1.png`
3. `optimized_model_performance.png`
4. `feature_ablation_macro_f1.png`
5. `cv_stability_macro_f1.png`
6. `optimized_error_transitions.png`
7. `decision_tree_feature_importance.png`
8. `google_play_vs_iarc_match.png`

## 15. 总体结论

当前实验已经形成完整闭环：

1. 数据规模满足要求，`1322` 条有效样本超过课程要求的 `1000` 条。
2. 已训练超过 3 种模型，实际完成 8 个第一版模型和 6 个优化模型。
3. 当前最佳模型为 `optimized_lightgbm`，holdout macro-F1 为 `0.872`，accuracy 为 `0.980`。
4. 特征消融表明问卷答案是主要信号，结果页派生统计提供额外增益。
5. CV 稳定性实验表明少数类导致评估波动，报告中应谨慎解释单次 holdout。
6. 错误案例主要是保守高估，没有观察到 `18+` 被严重低估到低龄类别。
7. 多机构评级分析显示 Google Play 与 IARC Generic 存在显示口径差异，可作为扩展讨论。
8. 地区评级预测任务进一步证明问卷特征可以迁移到多个评级机构标签，ESRB、Google Play 和 IARC Generic 均取得较高 macro-F1。

当前实验部分已经足够支撑完整课程报告。后续主要工作应转向报告写作、图表整理和结论表达。

最终建议：

```text
实验部分到这里可以收束。继续新增实验的边际收益较低，下一阶段应优先完成报告正文、图表排版和结论表述。
```
