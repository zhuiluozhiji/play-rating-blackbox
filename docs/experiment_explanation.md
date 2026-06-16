# Google Play 年龄分级黑盒逆向建模实验讲解

本文用于解释当前实验的研究目标、数据、特征工程、模型、评估指标、参数设置和第一版结果。

## 1. 实验目标

本实验把 Google Play / IARC 年龄分级问卷视为一个未知黑盒函数。开发者填写问卷答案后，Google Play Console 会返回最终年龄分级，但具体决策规则不公开。

因此，我们通过采集大量“问卷答案 - 评级结果”样本，训练一个替代模型来近似这个黑盒函数。

```text
黑盒真实规则:        f(x) -> y
我们训练的模型:      f_hat(x) -> y_hat
目标:               让 y_hat 尽量接近 y，并分析 f_hat 学到的关键规则
```

其中：

```text
x: 一条问卷回答向量
y: Google Play / IARC 返回的年龄分级
f: 真实但不可见的黑盒评级规则
f_hat: 我们训练得到的机器学习模型
```

本实验的核心问题是：能否仅根据问卷答案预测最终年龄分级，并进一步分析哪些问题最影响分级结果。

## 2. 当前数据集

当前主数据集是 `real_20260615_full`，有效样本数为 `1322`。主标签暂时使用 `IARC Generic` 口径，也就是标准样本字段 `result_age_rating`。

标签分布如下：

| 年龄分级 | 样本数 |
|---|---:|
| `3+` | 37 |
| `7+` | 19 |
| `12+` | 225 |
| `16+` | 28 |
| `18+` | 1013 |

可以看到，数据集高度不均衡。`18+` 占 `1013 / 1322 = 76.6%`，所以模型如果永远预测 `18+`，也能获得较高 accuracy。因此本实验不能只看 accuracy，必须重点看 macro-F1、balanced accuracy、少数类 recall 和严重错误率。

当前特征矩阵为：

```text
X shape = 1322 x 1010
```

`data/processed/real_20260615_full.features.csv` 的形状是 `1322 x 1011`，其中 1 列是标签，因此真正用于训练的输入特征数是 `1010`。

当前配置使用分层训练测试划分：

```text
random_seed = 42
test_size = 0.15
train size = 1123
test size = 199
```

测试集标签 support 为：

| 年龄分级 | 测试集样本数 |
|---|---:|
| `3+` | 6 |
| `7+` | 3 |
| `12+` | 34 |
| `16+` | 4 |
| `18+` | 152 |

这里要特别注意：`16+` 在测试集里只有 4 条，所以它的 precision、recall、F1 很容易受单个样本影响。

## 3. 特征工程

每条原始样本中保留完整问卷答案字段 `answers_json`。在建模前，脚本会将问卷答案展开为表格特征。

例如某个问题 `q_xxx` 回答 `Yes`，会产生 one-hot 特征：

```text
answer__q_xxx_Yes = 1
answer__q_xxx_No  = 0
```

如果一个问题是多选题，原始值会先按选项排序并用 `|` 连接，然后再进行类别编码。

除了问卷答案 one-hot 特征，当前框架还保留了一些统计特征：

```text
visible_question_count
skipped_question_count
content_descriptor_count
interactive_element_count
high_risk_count
medium_risk_count
triggered_branch_count
```

这些特征可以帮助模型捕捉问卷路径长度、触发分支数量、内容描述符数量和互动元素数量等宏观信息。

### CSV 读取细节

当前脚本读取 `dataset.csv` 时使用：

```python
pd.read_csv(path, keep_default_na=False, na_values=[""])
```

原因是 pandas 默认会把字符串 `"None"` 当作缺失值。但在问卷中，`None` 可能是一个合法选项，例如某些严重程度题目的选项。因此我们需要：

```text
空单元格 -> 仍然作为缺失值
字符串 "None" -> 保留为合法类别
```

这个细节会影响最终特征维度和模型训练结果。

## 4. 模型设置

当前配置文件为 `configs/modeling.yaml`，模型训练入口为 `scripts/train_models.py`。

当前共训练 8 个模型：

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

### 4.1 Majority Baseline

`majority` 是多数类基线模型，永远预测训练集中出现最多的类别。

公式为：

```text
y_hat = argmax_k count(y = k)
```

在当前数据中，多数类是 `18+`。这个模型用于证明：如果只看 accuracy，模型可能只是利用类别不均衡，并没有真正学到问卷规则。

### 4.2 Stratified Baseline

`stratified` 是按训练集标签分布随机预测的基线模型。

例如训练集中 `18+` 占比很高，那么该模型也会以较高概率预测 `18+`。

它的作用是提供一个随机但考虑类别分布的下限参考。

### 4.3 Logistic Regression

当前逻辑回归参数为：

```text
StandardScaler(with_mean=False)
LogisticRegression(
  max_iter=1000,
  class_weight="balanced",
  solver="liblinear"
)
```

由于 `solver="liblinear"`，多分类任务会以 one-vs-rest 方式处理。对每个类别 `k`，训练一个二分类器：

```text
P(y = k | x) = sigmoid(w_k^T x + b_k)
```

预测时选择概率最大的类别：

```text
y_hat = argmax_k P(y = k | x)
```

`class_weight="balanced"` 会提高少数类权重。类别权重近似为：

```text
weight_k = n_samples / (n_classes * n_samples_k)
```

其中 `n_samples_k` 是类别 `k` 的样本数。样本越少，权重越大。

`StandardScaler(with_mean=False)` 表示对特征做尺度缩放，但不做均值中心化。这适合 one-hot / 稀疏特征。

### 4.4 Decision Tree

当前决策树参数为：

```text
DecisionTreeClassifier(
  max_depth=8,
  min_samples_leaf=3,
  class_weight="balanced",
  random_state=42
)
```

决策树通过不断选择特征切分数据，使子节点中的类别更纯。

默认使用 Gini impurity：

```text
Gini(S) = 1 - sum_k p_k^2
```

其中 `p_k` 是节点 `S` 中类别 `k` 的比例。

每次切分希望最大化 impurity reduction：

```text
Gain = Gini(parent) - weighted_avg(Gini(children))
```

`max_depth=8` 限制树深，防止完全记忆训练集。`min_samples_leaf=3` 要求叶子节点至少有 3 个样本，也用于抑制过拟合。

### 4.5 Random Forest

当前随机森林参数为：

```text
RandomForestClassifier(
  n_estimators=300,
  min_samples_leaf=2,
  class_weight="balanced",
  random_state=42,
  n_jobs=-1
)
```

随机森林训练 300 棵决策树，每棵树基于 bootstrap 样本和随机特征子集训练。最终预测使用投票：

```text
y_hat = majority_vote(T_1(x), T_2(x), ..., T_300(x))
```

随机森林通常比单棵树更稳定，因为多棵树平均后能降低方差。

### 4.6 Extra Trees

当前 Extra Trees 参数为：

```text
ExtraTreesClassifier(
  n_estimators=300,
  min_samples_leaf=2,
  class_weight="balanced",
  random_state=42,
  n_jobs=-1
)
```

Extra Trees 和随机森林类似，也训练多棵树。但它在选择切分点时引入更多随机性，通常能进一步降低方差，在高维 one-hot 特征上经常表现不错。

### 4.7 XGBoost

当前 XGBoost 参数为：

```text
XGBClassifier(
  n_estimators=200,
  max_depth=4,
  learning_rate=0.05,
  subsample=0.9,
  eval_metric="mlogloss",
  verbosity=0,
  random_state=42
)
```

XGBoost 是梯度提升树模型。它不是一次训练很多独立树，而是一轮一轮地加树，每一轮拟合前一轮的残差或梯度方向。

形式上可以写成：

```text
F_t(x) = F_{t-1}(x) + eta * tree_t(x)
```

其中：

```text
F_t: 第 t 轮后的模型
eta: learning_rate，这里是 0.05
tree_t: 第 t 轮新增的树
```

多分类时主要优化 multiclass log loss：

```text
Loss = - sum_i log P(y_i | x_i)
```

`max_depth=4` 表示每棵树较浅，`n_estimators=200` 表示最多叠加 200 棵树，`subsample=0.9` 表示每轮使用 90% 样本，增加随机性并减少过拟合。

### 4.8 LightGBM

当前 LightGBM 参数为：

```text
LGBMClassifier(
  n_estimators=200,
  max_depth=5,
  learning_rate=0.05,
  subsample=0.9,
  force_row_wise=True,
  verbosity=-1,
  random_state=42
)
```

LightGBM 也是梯度提升树模型，但实现上更强调速度和高维特征效率。它使用直方图算法，并倾向于 leaf-wise 生长。

核心思想与 XGBoost 类似：

```text
F_t(x) = F_{t-1}(x) + eta * tree_t(x)
```

`verbosity=-1` 用于关闭大量训练日志，`force_row_wise=True` 用于稳定当前 Windows 环境下的训练输出。

## 5. 评估指标

### 5.1 Accuracy

Accuracy 表示预测正确的样本比例：

```text
Accuracy = (1 / n) * sum_i 1[y_i = y_hat_i]
```

但在当前数据集中，`18+` 占比过高，所以 accuracy 可能虚高。

### 5.2 Precision、Recall、F1

对每个类别 `k`：

```text
Precision_k = TP_k / (TP_k + FP_k)
Recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 * Precision_k * Recall_k / (Precision_k + Recall_k)
```

其中：

```text
TP_k: 真实为 k，预测也为 k
FP_k: 真实不是 k，但预测为 k
FN_k: 真实为 k，但预测成其他类别
```

### 5.3 Macro-F1

Macro-F1 对每个类别的 F1 取简单平均：

```text
Macro-F1 = (1 / K) * sum_k F1_k
```

它不会因为 `18+` 样本多就给 `18+` 更大权重，因此更适合当前不均衡数据。

### 5.4 Weighted-F1

Weighted-F1 按每个类别样本数加权：

```text
Weighted-F1 = sum_k (n_k / n) * F1_k
```

它比 macro-F1 更受多数类影响，但比 accuracy 更细。

### 5.5 Balanced Accuracy

Balanced accuracy 是各类 recall 的平均：

```text
Balanced Accuracy = (1 / K) * sum_k Recall_k
```

它关注每个类别是否都能被召回。

### 5.6 Mean Absolute Age-Level Error

年龄分级是有序类别，因此我们定义顺序：

```text
3+  -> 0
7+  -> 1
12+ -> 2
16+ -> 3
18+ -> 4
```

平均年龄等级误差为：

```text
MAE = (1 / n) * sum_i |order(y_i) - order(y_hat_i)|
```

例如真实 `18+` 预测为 `16+`，误差是 `1`。真实 `18+` 预测为 `12+`，误差是 `2`。

### 5.7 Severe Error Rate

严重错误定义为预测等级与真实等级相差至少 2 档：

```text
Severe Error Rate = (1 / n) * sum_i 1[ |order(y_i) - order(y_hat_i)| >= 2 ]
```

例如：

```text
18+ -> 12+ 是严重错误
16+ -> 7+ 是严重错误
12+ -> 18+ 是严重错误
```

该指标适合年龄分级任务，因为把高风险内容低估为低龄分级，比相邻类别混淆更严重。

## 6. 第一版模型结果

当前第一版模型指标如下：

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

从结果看：

```text
accuracy 最高: lightgbm = 0.975
macro-F1 最高: xgboost = 0.833
severe error 最低: lightgbm = 0.010
```

由于当前数据类别不均衡，报告中更应该强调 `macro-F1`，因此第一版主模型可以选择 `xgboost`。

## 7. XGBoost 逐类别结果

`xgboost` 是当前 macro-F1 最好的模型。它在测试集上的逐类别表现为：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `3+` | 1.000 | 0.667 | 0.800 | 6 |
| `7+` | 1.000 | 1.000 | 1.000 | 3 |
| `12+` | 1.000 | 0.971 | 0.985 | 34 |
| `16+` | 1.000 | 0.250 | 0.400 | 4 |
| `18+` | 0.962 | 1.000 | 0.981 | 152 |

这里最重要的问题是 `16+`。它的 precision 是 `1.000`，说明模型一旦预测 `16+` 基本是对的。但 recall 只有 `0.250`，说明真实 `16+` 中只有 1/4 被找出来，其余被预测成其他类别。

原因主要有两个：

```text
16+ 总样本数只有 28
测试集中 16+ support 只有 4
```

所以 `16+` 的指标不稳定，后续如果要优化，应重点补充或重采样 `16+` 边界样本。

## 8. 可解释性分析

当前解释性分析输出在：

```text
outputs/analysis/current/explanations/
```

### 8.1 树模型特征重要性

对于 `decision_tree`、`random_forest`、`extra_trees`、`xgboost`、`lightgbm`，当前脚本会输出模型原生的 `feature_importances_`。

对于 sklearn 树模型，特征重要性可以理解为该特征在所有树分裂中带来的 impurity reduction 总贡献。

简化公式为：

```text
Importance_j = sum over splits using feature j of weighted impurity decrease
```

它回答的问题是：哪些问卷选项最常被模型用于降低分类不确定性。

### 8.2 决策树规则

单棵 `decision_tree` 还会导出规则文本：

```text
outputs/analysis/current/explanations/decision_tree_rules.txt
```

这类规则适合放进报告，用来展示模型如何近似黑盒判断路径。

### 8.3 Permutation Importance

逻辑回归没有原生树重要性，因此当前对 `logistic_regression` 使用 permutation importance。

公式为：

```text
Importance_j = Score(X, y) - Score(permute_column_j(X), y)
```

也就是打乱第 `j` 个特征后，如果模型分数明显下降，则说明该特征重要。

当前解释脚本使用参数：

```text
MAX_PERMUTATION_SAMPLES = 200
PERMUTATION_REPEATS = 1
```

这样可以避免在 1000 多个特征上反复打乱造成超时。

## 9. 图表输出

当前图表输出在：

```text
outputs/analysis/current/figures/
```

主要包括：

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

这些图可以直接用于实验报告：

```text
label_distribution.png: 展示类别不均衡
model_performance.png: 展示不同模型性能对比
confusion_matrix_xgboost.png: 展示最佳 macro-F1 模型的错误分布
*_feature_importance.png: 展示关键问卷特征
```

## 10. 当前实验结论

第一版实验已经说明：

1. Google Play / IARC 年龄分级虽然是黑盒，但问卷答案与最终分级之间存在明显可学习映射。
2. 复杂树模型明显优于多数类基线，说明模型不是单纯依赖 `18+` 多数类。
3. `xgboost` 当前 macro-F1 最好，适合作为第一版主模型。
4. `lightgbm` 的 accuracy 和 severe error rate 最好，可以作为对照模型。
5. `16+` 是当前最薄弱类别，主要受样本少和边界模糊影响。
6. 报告中应优先使用 macro-F1、balanced accuracy、per-class recall 和 severe error rate，而不是只使用 accuracy。

## 11. 后续优化方向

后续如果继续完善，可以做以下几件事：

1. 使用 Stratified K-fold cross validation，降低单次 holdout 带来的偶然性。
2. 针对 `16+` 做少数类补采或边界样本增强。
3. 加入更系统的超参数搜索，例如 grid search 或 random search。
4. 对 `result_region_ratings` 做扩展任务，比较 Google Play、IARC、ESRB、PEGI 等机构分级差异。
5. 增加反事实分析，例如只改变某个问卷选项，观察预测等级是否跳变。
6. 将特征重要性排名映射回原始问题文本，使报告解释更直观。

## 12. 当前产物位置

关键产物如下：

```text
outputs/analysis/current/metrics/model_metrics.json
outputs/analysis/current/models/
outputs/analysis/current/explanations/
outputs/analysis/current/figures/
outputs/analysis/current/metrics/region_rating_summary.json
```

当前测试状态：

```text
pytest: 4 passed
```

