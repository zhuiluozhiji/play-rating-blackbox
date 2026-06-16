# 地区评级预测任务实验汇总

本文记录“地区评级预测任务”的实现、模型设置、输出文件和第一版结果。该任务基于现有 full 数据集完成，没有新增数据采集。

## 1. 任务定义

主任务使用 `result_age_rating` 预测 IARC Generic 年龄标签。本扩展任务进一步利用每条样本中的多机构评级字段：

```text
result_region_ratings
```

对不同评级机构分别建立预测模型。也就是说，同一份问卷答案特征可以对应多个地区/机构标签，例如：

```text
Google Play
IARC Generic
Entertainment Software Rating Board (ESRB)
Pan-European Game Information (PEGI)
Unterhaltungssoftware Selbstkontrolle (USK)
Classificação Indicativa (ClassInd)
Australian Classification Board (ACB)
Digital Game Self-regulation Committee (DGSC)
```

该任务回答的问题是：

```text
给定一份 IARC / Google Play 年龄评级问卷答案，模型能否同时近似不同地区评级机构的标签映射？
```

## 2. 数据与标签

使用数据文件：

```text
data/raw/real_20260615_full.samples.jsonl
```

多机构评级统计文件：

```text
outputs/analysis/current/metrics/region_rating_summary.json
```

数据覆盖情况：

```text
total_success_samples = 1322
samples_with_region_ratings = 1322
missing_region_ratings = 0
distinct_authority_count = 10
```

各机构原始标签分布已经保存在：

```text
outputs/analysis/current/metrics/region_rating_summary.json
```

其中有 8 个机构满足本轮训练条件，2 个机构被跳过：

| 机构 | 样本数 | 类别数 | 最小类别样本数 | 状态 |
|---|---:|---:|---:|---|
| IARC Generic | 1322 | 5 | 19 | 已训练 |
| ESRB | 1322 | 5 | 16 | 已训练 |
| Google Play | 1322 | 6 | 19 | 已训练 |
| PEGI | 1322 | 6 | 19 | 已训练 |
| USK | 1322 | 5 | 11 | 已训练 |
| ClassInd | 1322 | 6 | 7 | 已训练 |
| DGSC | 514 | 5 | 3 | 已训练 |
| ACB | 514 | 6 | 3 | 已训练 |
| GRAC | 514 | 4 | 2 | 跳过 |
| Gmedia | 514 | 6 | 1 | 跳过 |

跳过规则：

```text
min_samples = 100
min_classes = 2
min_class_count = 3
```

GRAC 和 Gmedia 的最小类别样本数低于 3，若强行训练，分层划分和少数类评价会非常不稳定，因此本轮只做统计，不报告模型指标。

## 3. 特征与防泄漏处理

地区评级预测仍复用主实验的问卷特征工程：

```text
answer__question_id_option one-hot 特征
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

关键防泄漏设置：

```text
训练地区评级模型时，显式移除 result_age_rating。
```

原因是 `result_age_rating` 本身就是 IARC Generic 主标签。如果把它作为特征输入，模型在预测 Google Play、PEGI、USK 等地区标签时会间接看到答案，导致指标虚高。因此地区任务只使用问卷答案和派生统计特征，不使用主标签作为输入。

## 4. 模型与参数

脚本：

```text
scripts/train_region_rating_models.py
```

模型：

```text
LightGBM multiclass classifier
```

使用主任务优化实验中表现最好的 LightGBM 参数作为地区任务统一基线：

```text
n_estimators = 300
max_depth = 5
learning_rate = 0.05
subsample = 0.9
num_leaves = 15
min_child_samples = 10
force_row_wise = True
verbosity = -1
random_state = 42
```

划分方式：

```text
test_size = 0.15
random_seed = 42
stratify = y
```

类别不均衡处理：

```text
sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
```

权重公式可以理解为：

```text
w_k = n / (K * n_k)
```

其中 `n` 是训练样本数，`K` 是类别数，`n_k` 是类别 `k` 的训练样本数。少数类样本获得更高权重，缓解模型只偏向多数类的问题。

## 5. 评估指标

地区评级标签之间不是统一的 `3+ / 7+ / 12+ / 16+ / 18+` 年龄序列，例如 ESRB 有 `Teen`、`Mature 17+`、`Adults only 18+`，ACB 和 DGSC 还包含 `Refused Classification`。因此本任务不计算主任务里的年龄等级 MAE 和 severe error，而使用普通多分类指标：

Accuracy：

```text
Accuracy = (1 / n) * sum_i 1[y_i = y_hat_i]
```

每类 F1：

```text
Precision_k = TP_k / (TP_k + FP_k)
Recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 * Precision_k * Recall_k / (Precision_k + Recall_k)
```

Macro-F1：

```text
Macro-F1 = (1 / K) * sum_k F1_k
```

Weighted-F1：

```text
Weighted-F1 = sum_k (n_k / n) * F1_k
```

Balanced accuracy：

```text
Balanced Accuracy = (1 / K) * sum_k Recall_k
```

对于不均衡地区评级任务，报告中应优先解释 `macro_f1` 和 `balanced_accuracy`，因为它们比 accuracy 更能反映少数类表现。

## 6. 输出文件

总指标：

```text
outputs/analysis/current/metrics/region_rating_model_metrics.json
outputs/analysis/current/metrics/region_rating_model_summary.csv
```

逐机构混淆矩阵：

```text
outputs/analysis/current/metrics/region_rating_confusion/
```

逐机构 holdout 预测明细：

```text
outputs/analysis/current/metrics/region_rating_predictions/
```

逐机构模型：

```text
outputs/analysis/current/models/region_rating_*.joblib
```

综合图表：

```text
outputs/analysis/current/figures/region_rating_prediction_performance.png
```

## 7. 第一版模型结果

| Authority | Samples | Classes | Min Class | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| IARC Generic | 1322 | 5 | 19 | 0.975 | 0.829 | 0.972 | 0.767 |
| ESRB | 1322 | 5 | 16 | 0.884 | 0.822 | 0.884 | 0.807 |
| Google Play | 1322 | 6 | 19 | 0.960 | 0.811 | 0.955 | 0.787 |
| PEGI | 1322 | 6 | 19 | 0.960 | 0.748 | 0.948 | 0.744 |
| USK | 1322 | 5 | 11 | 0.970 | 0.738 | 0.965 | 0.749 |
| ClassInd | 1322 | 6 | 7 | 0.915 | 0.674 | 0.913 | 0.691 |
| DGSC | 514 | 5 | 3 | 0.897 | 0.585 | 0.897 | 0.676 |
| ACB | 514 | 6 | 3 | 0.910 | 0.522 | 0.904 | 0.516 |

跳过机构：

| Authority | Samples | Classes | Min Class | Reason |
|---|---:|---:|---:|---|
| GRAC | 514 | 4 | 2 | `min_class_count < 3` |
| Gmedia | 514 | 6 | 1 | `min_class_count < 3` |

## 8. 结果解读

ESRB 是本轮地区评级任务中 macro-F1 最高的机构：

```text
accuracy = 0.884
macro_f1 = 0.822
balanced_accuracy = 0.807
```

这说明 ESRB 虽然总体 accuracy 不如 IARC Generic、Google Play、PEGI 和 USK，但不同类别之间识别更均衡。

Google Play 与 IARC Generic 都取得较高 accuracy：

```text
Google Play accuracy = 0.960, macro_f1 = 0.811
IARC Generic accuracy = 0.975, macro_f1 = 0.829
```

这里的 IARC Generic 是地区任务中的 sanity check，使用统一的地区任务 LightGBM 设置；它不替代主任务中已经单独优化得到的 `optimized_lightgbm` 最佳结果：

```text
optimized_lightgbm main task macro_f1 = 0.872
region-task IARC Generic macro_f1 = 0.829
```

Google Play 标签中存在 `Rated for 19+`，因此它不是 IARC Generic 的简单复制。前置统计中已经发现：

```text
Google Play vs IARC Generic exact_match_count = 748
Google Play vs IARC Generic mismatch_count = 574
```

PEGI 和 USK 的 accuracy 很高，但 macro-F1 低于 Google Play 和 ESRB：

```text
PEGI macro_f1 = 0.748
USK macro_f1 = 0.738
```

主要原因是少数类 support 很小，例如 PEGI 16 在测试集中只有 4 条，USK 6+ 在测试集中只有 2 条，这会显著压低 macro-F1。

ACB 和 DGSC 的 accuracy 看起来不低，但 macro-F1 明显较低：

```text
ACB macro_f1 = 0.522
DGSC macro_f1 = 0.585
```

原因是这两个机构只有 514 条样本，并且包含 `Refused Classification` 这类强多数类。模型能较好识别多数类，但对极少数类别不稳定。

## 9. 典型错误方向

ACB 的主要错误包括：

```text
Restricted to 18+ -> Refused Classification: 2
Refused Classification -> Restricted to 15+: 2
Restricted to 15+ -> Mature: 1
Mature -> Restricted to 15+: 1
General -> Refused Classification: 1
```

ClassInd 的主要错误集中在相邻或近似年龄段：

```text
Rated 14+ -> Rated 16+: 6
Rated 18+ -> Rated 14+: 2
Rated 16+ -> Rated 18+: 2
```

ESRB 的主要错误集中在高年龄段之间：

```text
Adults only 18+ -> Mature 17+: 8
Teen -> Mature 17+: 5
Adults only 18+ -> Teen: 5
```

Google Play 的错误数量较少，但少数类仍不稳定：

```text
Rated for 3+ -> Rated for 19+: 2
Rated for 3+ -> Rated for 12+: 1
Rated for 19+ -> Rated for 12+: 1
Rated for 18+ -> Rated for 16+: 1
Rated for 16+ -> Rated for 3+: 1
```

IARC Generic 的错误主要来自 `16+` 和低龄少数类：

```text
Rated for 16+ -> Rated for 18+: 2
Rated for 7+ -> Rated for 12+: 1
Rated for 3+ -> Rated for 18+: 1
Rated for 3+ -> Rated for 16+: 1
```

## 10. 可以写进报告的结论

地区评级预测任务证明，当前采集的问卷特征不仅能预测主标签 IARC Generic，也能较好预测 Google Play、ESRB、PEGI、USK、ClassInd 等地区/机构标签。

最适合写进报告的结论是：

```text
在不使用 result_age_rating 作为输入特征的前提下，LightGBM 仍能对多个地区评级机构取得较高预测性能。ESRB 的 macro-F1 达到 0.822，Google Play 的 macro-F1 达到 0.811，IARC Generic 的 macro-F1 达到 0.829。这表明问卷答案中包含可迁移到不同地区评级体系的核心信号。
```

也需要同时说明局限性：

```text
ACB、DGSC、GRAC 和 Gmedia 等机构的可用样本数较少或类别极不均衡，因此地区任务的少数类指标仍存在较大不确定性。尤其当最小类别样本数低于 3 时，本实验选择跳过模型训练，只保留统计分析。
```

## 11. 与主任务的关系

主任务仍然建议保持为：

```text
IARC Generic / result_age_rating 多分类预测
```

地区评级预测任务适合作为扩展实验，放在报告的后半部分。它的作用是：

```text
1. 证明当前数据中多机构标签没有浪费，可以支持更丰富的分析。
2. 展示不同评级体系之间既有共性，也有口径差异。
3. 用 Google Play 与 IARC Generic 的 574 条不一致样本解释为什么主标签需要固定口径。
4. 为后续研究留下 multi-task learning 或 label mapping 的扩展方向。
```

## 12. 复现实验命令

训练地区评级模型：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\train_region_rating_models.py
```

生成综合图表：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\make_experiment_figures.py
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
