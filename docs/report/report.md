# Google Play 年龄分级问卷的黑盒逆向建模与预测

## 摘要

Google Play 年龄分级问卷的具体决策规则并未公开，本文将其视为黑盒系统，研究问卷答案能否预测最终年龄分级。实验针对树状条件问卷设计了自动化采集流程，通过浏览器自动探索有效问卷路径、解析结果页、按答案签名去重，并结合少数类补采构建真实样本集。最终数据集包含 `1322` 条有效样本，主标签为 `IARC Generic` 年龄分级，特征矩阵规模为 `1322 x 1010`。

本文训练并比较了 baseline、Logistic Regression、Decision Tree、Random Forest、XGBoost 和 LightGBM 等多分类模型。优化后的 LightGBM 在 holdout test 上取得 `0.980` accuracy、`0.872` macro-F1 和 `0.010` severe error rate；answer-only 消融实验仍达到 `0.774` macro-F1，说明问卷答案本身已包含较强预测信号。进一步分析表明，结果页描述符和互动元素可增强黑盒解释能力，模型错误主要表现为保守高估，未观察到对 `18+` 样本的严重低估。实验同时发现，类别不均衡和少数类样本不足仍是模型稳定性的主要限制。

---

## 1. 引言与问题定义

### 1.1 研究背景

年龄分级用于帮助家长和用户判断应用是否适合未成年人。在 Google Play 中，开发者需要填写内容分级问卷，系统随后生成年龄分级、内容描述符以及互动元素标签。这些结果会出现在 Google Play 商店页面中，影响用户对应用内容风险的判断。

然而，Google Play 并未公开问卷答案到最终评级之间的完整决策逻辑。对于开发者和普通用户而言，该过程可以被视为一个典型黑盒系统：我们可以观察输入和输出，却无法直接获得内部规则。

### 1.2 问题定义

本文将实验任务形式化为多分类预测问题：

```text
输入：Google Play / IARC 内容分级问卷答案
输出：最终 IARC Generic 年龄分级
标签：3+、7+、12+、16+、18+
```

实验目标不是声称恢复官方规则，而是构建一个数据驱动的替代模型，用以近似黑盒输入输出行为，并分析哪些问卷特征可能对评级结果更重要。

### 1.3 实验贡献

本项目的亮点主要体现在四个方面：

1. **真实黑盒采集**：通过自动化浏览器流程采集真实问卷结果，而非构造模拟数据。
2. **树状问卷探索**：针对条件跳转问卷实现路径采样、去重和少数类补采。
3. **系统性建模评估**：比较多类模型，并使用 macro-F1、balanced accuracy 和 severe error rate 处理类别不均衡。
4. **机制解释扩展**：通过特征重要性、消融实验、反事实翻转、区域评级预测等方法分析黑盒机制。

---

## 2. 实验系统设计与自动化采集

### 2.1 总体实验流程

本实验从真实问卷采集开始，经过样本转换、特征工程和模型训练，最终输出预测结果与解释分析。整体流程可以概括为：首先自动探索内容分级问卷的有效路径，提交后解析评级结果；随后对样本进行去重、清洗和结构化编码；最后训练多种分类模型，并通过消融、错误分析和特征解释评估黑盒替代模型的可靠性。

![整体实验流程图](<../figure/image copy 2.png>)

### 2.2 树状问卷自动化采集

Google Play 内容分级问卷不是固定表单。后续问题会根据前序答案动态出现，因此采集脚本需要在每一步识别当前可见问题，再选择答案并继续推进问卷。

![树状问卷探索流程图](<../figure/image copy.png>)

本项目使用基于 Chrome DevTools Protocol 和 Playwright 的自动化流程。脚本连接到已登录的浏览器实例，自动选择问卷选项、提交问卷、解析评级结果，并将样本增量保存为 JSONL 文件。

核心采集流程如下：

```python
for i in range(sample_count):
    page = connect_to_logged_in_chrome()
    answers = {}

    while questionnaire_can_continue(page):
        questions = parse_visible_questions(page)
        question = choose_next_question(questions)
        option = sample_option(question)
        select_option(page, question, option)
        answers[question.id] = option
        click_next_or_save(page)

    result = parse_rating_result(page)
    signature = hash_answers(answers)

    if signature not in seen_signatures:
        append_jsonl(answers, result, signature)
```

### 2.3 采样与去重策略

采集策略以随机路径探索为主，并配合少数类补采。由于随机选择问卷路径很容易触发高风险选项，初始数据中 `18+` 样本占比较高。因此项目后续加入低风险和中风险路径补采，用于提高 `3+`、`7+`、`12+`、`16+` 的样本覆盖。

采集阶段使用以下机制保证数据质量：

| 机制 | 作用 |
|---|---|
| 随机路径探索 | 覆盖树状问卷的不同路径 |
| `response_signature` | 对答案组合哈希去重 |
| 增量 JSONL 写入 | 避免浏览器崩溃导致数据丢失 |
| resume 支持 | 允许中断后继续采集 |
| loop detection | 识别异常循环状态 |
| 少数类补采 | 缓解类别不均衡 |

在实现层面，采集系统围绕“识别状态—选择答案—推进问卷—解析结果”循环运行：脚本首先连接已登录浏览器并识别当前问卷状态，然后基于可见问题选择合法答案并推进页面，最后解析结果页，将答案、评级和元信息保存为结构化样本。

---

## 3. 数据集构建与统计分析

### 3.1 数据组织

最终数据由三层组成：第一层是原始问卷样本，保留每条路径的完整答案和评级结果；第二层是清洗后的结构化数据，用于统计分析和标签检查；第三层是面向机器学习模型的特征矩阵。这样的分层设计既保留了可追溯性，也便于后续建模和复现实验。

主标签字段为：

```text
result_age_rating
```

对应口径为 `IARC Generic` 年龄分级。

### 3.2 数据规模与质量

最终数据集统计如下：

| 指标 | 数值 |
|---|---:|
| 输入记录数 | `1341` |
| 成功完成记录数 | `1339` |
| loop-detected 记录数 | `2` |
| 去重跳过记录数 | `17` |
| 有效去重样本数 | `1322` |
| 无效样本数 | `0` |
| 缺失标签数 | `0` |
| 唯一答案组合数 | `1322` |
| 发现问题数 | `229` |

数据集规模超过实验要求中的 1000 条有效样本。所有最终样本均包含完整答案、评级标签和解析后的评级结果。

### 3.3 标签分布

最终标签分布如下：

| 年龄分级 | 样本数 | 占比 |
|---|---:|---:|
| `3+` | 37 | 2.8% |
| `7+` | 19 | 1.4% |
| `12+` | 225 | 17.0% |
| `16+` | 28 | 2.1% |
| `18+` | 1013 | 76.6% |

![标签分布](../../outputs/analysis/current/figures/experiment_label_distribution.png)

可以看到，数据集存在明显类别不均衡。`18+` 类别占比超过四分之三，若模型始终预测 `18+`，也能获得较高 accuracy。因此后续评估不能只依赖 accuracy，而必须同时关注 macro-F1、balanced accuracy 和 severe error rate。

### 3.4 数据质量控制

实验通过以下步骤控制数据质量：

- 使用答案签名去除重复路径。
- 过滤未完成问卷和循环异常样本。
- 检查 `result_age_rating` 是否缺失。
- 保留完整答案字典，便于后续复查。
- 将原始样本、处理后数据和特征矩阵分层保存。

---

## 4. 特征工程与实验设置

### 4.1 问卷答案编码

每条样本保存完整问卷答案。建模时，问卷答案被展开为 one-hot 特征：

```text
answer__question_id_option = 0 or 1
```

这种表示方式保留了具体问题和具体选项，适合树模型、线性模型和特征重要性分析。

### 4.2 派生统计特征

除答案特征外，实验还构造了统计和聚合特征：

| 特征 | 含义 |
|---|---|
| `visible_question_count` | 当前路径中出现的问题数 |
| `skipped_question_count` | 条件跳转导致跳过的问题数 |
| `content_descriptor_count` | 结果页内容描述符数量 |
| `interactive_element_count` | 结果页互动元素数量 |
| `violence_score` 等 | 各风险主题的聚合得分 |
| `high_risk_count` | 高风险主题计数 |
| `medium_risk_count` | 中风险主题计数 |
| `triggered_branch_count` | 被触发的问卷分支数量 |

最终特征矩阵规模为：

```text
X shape = 1322 x 1010
feature matrix with label = 1322 x 1011
```

其中多出的 1 列是目标标签。

### 4.3 信息泄漏控制

本项目区分两种预测设置：

| 设置 | 特征范围 | 用途 |
|---|---|---|
| Full-feature | 问卷答案 + 结果页统计特征 | 最强黑盒替代模型与机制分析 |
| Answer-only | 仅问卷答案 | 更接近提交前预测场景 |

![特征工程与泄漏控制图](../figure/image.png)

`content_descriptor_count` 和 `interactive_element_count` 来自结果页，严格来说不是提交前可用信息。因此本文不仅报告 full-feature 模型，也通过 answer-only 消融实验评估问卷答案本身的预测能力。这一点可以避免将后验信息误认为预提交预测能力。

换言之，answer-only 设置回答的是“提交问卷前，仅凭答案能否预测评级”；full-feature 设置回答的是“从完整输入输出行为出发，哪些结果结构与评级最相关”。二者关注点不同，因此本文在解释结果时分别讨论，避免夸大模型的实际预提交能力。

### 4.4 实验设置与指标

训练测试划分如下：

| 设置 | 数值 |
|---|---:|
| 随机种子 | `42` |
| 测试集比例 | `0.15` |
| 训练集大小 | `1123` |
| 测试集大小 | `199` |

测试集类别 support：

| 年龄分级 | 测试样本数 |
|---|---:|
| `3+` | 6 |
| `7+` | 3 |
| `12+` | 34 |
| `16+` | 4 |
| `18+` | 152 |

由于少数类样本较少，本文同时报告：

- Accuracy
- Macro-F1
- Balanced Accuracy
- Weighted-F1
- Mean Absolute Age Error
- Severe Error Rate

其中 severe error 定义为预测等级与真实等级相差至少 2 级。

---

## 5. 模型训练与性能评估

### 5.1 模型列表

实验比较了以下模型：

| 模型 | 作用 |
|---|---|
| Majority baseline | 多数类基线 |
| Stratified baseline | 按标签分布随机预测 |
| Logistic Regression | 可解释线性模型 |
| Decision Tree | 规则近似模型 |
| Random Forest | 集成树模型 |
| Extra Trees | 随机化树集成 |
| XGBoost | 梯度提升树 |
| LightGBM | 高效梯度提升树 |

核心训练流程如下：

```python
samples = load_valid_questionnaire_samples()
dataset = build_dataset(samples)
X, y = build_features(dataset, label="result_age_rating")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = evaluate_multiclass_rating(y_test, pred)
```

### 5.2 基础模型对比

第一版模型对比如下：

| 模型 | Accuracy | Macro-F1 | Balanced Acc | Weighted-F1 | MAE | Severe Error |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.970 | 0.833 | 0.777 | 0.965 | 0.065 | 0.015 |
| LightGBM | 0.975 | 0.824 | 0.783 | 0.971 | 0.050 | 0.010 |
| Extra Trees | 0.955 | 0.745 | 0.716 | 0.948 | 0.095 | 0.025 |
| Logistic Regression | 0.970 | 0.748 | 0.765 | 0.960 | 0.070 | 0.015 |
| Random Forest | 0.910 | 0.642 | 0.667 | 0.900 | 0.171 | 0.060 |
| Decision Tree | 0.915 | 0.608 | 0.642 | 0.908 | 0.166 | 0.065 |

![基础模型性能](../../outputs/analysis/current/figures/model_performance.png)

所有正式模型均明显优于 baseline。第一版中，XGBoost 的 macro-F1 最高，LightGBM 的 accuracy 和 severe error rate 更优。

### 5.3 模型优化结果

进一步调参后，optimized LightGBM 成为综合表现最好的模型：

| 模型 | Accuracy | Macro-F1 | Balanced Acc | Severe Error |
|---|---:|---:|---:|---:|
| Optimized LightGBM | 0.980 | 0.872 | 0.833 | 0.010 |
| Optimized XGBoost | 0.955 | 0.776 | 0.778 | 0.025 |
| Optimized Extra Trees | 0.955 | 0.745 | 0.716 | 0.025 |
| Optimized Logistic Regression | 0.965 | 0.745 | 0.764 | 0.020 |
| Optimized Decision Tree | 0.925 | 0.735 | 0.708 | 0.035 |
| Optimized Random Forest | 0.950 | 0.718 | 0.682 | 0.030 |

![优化前后 Macro-F1](../../outputs/analysis/current/figures/base_vs_optimized_macro_f1.png)

LightGBM 的 macro-F1 从 `0.824` 提升到 `0.872`，同时保持最低 severe error rate。该结果说明，梯度提升树适合处理本实验中的高维稀疏问卷特征。

### 5.4 最佳模型逐类别表现

optimized LightGBM 的逐类别结果如下：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `3+` | 1.000 | 0.667 | 0.800 | 6 |
| `7+` | 1.000 | 1.000 | 1.000 | 3 |
| `12+` | 0.971 | 1.000 | 0.986 | 34 |
| `16+` | 1.000 | 0.500 | 0.667 | 4 |
| `18+` | 0.987 | 1.000 | 0.993 | 152 |

模型在 `12+` 和 `18+` 上表现稳定，`16+` 是最难类别。主要原因是 `16+` 总样本数仅 28 条，测试集中只有 4 条，少量边界样本就会显著影响该类指标。

---

## 6. 结果分析与黑盒机制解释

### 6.1 错误方向分析

optimized LightGBM 在 199 条测试样本中仅产生 4 个错误。错误主要表现为保守高估，例如将 `16+` 预测为 `18+`。值得注意的是，模型没有将 `18+` 严重低估为低龄类别。

![错误转移](../../outputs/analysis/current/figures/optimized_error_transitions.png)

对于年龄分级任务，低估高风险内容通常更严重。当前模型的错误方向更偏向保守，但对低风险应用的过度高估仍可能影响应用展示和用户认知。

### 6.2 特征重要性

特征重要性分析显示，问卷路径长度、互动元素数量、内容描述符数量以及若干具体问卷答案具有较强预测作用。

![LightGBM 特征重要性](../../outputs/analysis/current/figures/lightgbm_feature_importance.png)

部分高重要特征可映射回真实问卷含义：

| 特征含义 | 解释 |
|---|---|
| Scary elements | 恐怖或惊吓元素与评级变化强相关 |
| App category | 应用类别会影响后续问卷路径 |
| Offensive language | 攻击性语言有助于区分中低与中高评级 |
| Age-restricted items | 年龄限制活动或商品是高风险信号 |
| visible question count | 路径越长通常表示触发了更多风险分支 |

这表明模型不是仅依赖标签分布，而是学习到了问卷路径和内容风险之间的结构性关系。

### 6.3 反事实特征翻转

反事实实验选取 optimized LightGBM 的高重要二元特征，将单个特征从 0 翻转为 1 或从 1 翻转为 0，观察预测是否变化。

![反事实预测转移](../../outputs/analysis/current/figures/counterfactual_prediction_transitions.png)

实验共测试 25 个高重要特征，发现 62 个预测发生变化的样本。主要变化包括：

| 预测变化 | 数量 |
|---|---:|
| `12+ -> 18+` | 24 |
| `16+ -> 18+` | 12 |
| `7+ -> 12+` | 7 |
| `18+ -> 16+` | 4 |
| `12+ -> 3+` | 4 |

该实验说明模型在部分关键特征上形成了明显局部边界。不过，单点特征翻转不一定对应真实合法问卷路径，因此该结果应解释为模型敏感性，而不是官方规则。

### 6.4 黑盒机制发现

综合模型结果和解释分析，可以得到以下观察：

1. 问卷答案对年龄分级具有强预测能力。
2. 触发更多问卷分支通常意味着内容风险更复杂。
3. 恐怖元素、攻击性语言、年龄限制活动等特征与评级变化关系明显。
4. `16+` 与 `18+` 的边界较难，容易出现保守高估。
5. 黑盒模型可以近似输入输出行为，但不能等同于官方决策规则。

---

## 7. 补充实验与扩展分析

### 7.1 特征消融实验

为了分析不同特征组的贡献，实验对 optimized LightGBM 进行特征消融：

| 特征集合 | 特征数 | Accuracy | Macro-F1 | Balanced Acc | Severe Error |
|---|---:|---:|---:|---:|---:|
| Full | 1010 | 0.980 | 0.872 | 0.833 | 0.010 |
| No strategy | 1009 | 0.980 | 0.872 | 0.833 | 0.010 |
| No aggregate scores | 999 | 0.980 | 0.872 | 0.833 | 0.010 |
| No descriptor/interactive | 1008 | 0.965 | 0.787 | 0.781 | 0.020 |
| Answer-only | 994 | 0.960 | 0.774 | 0.811 | 0.025 |
| Counts and scores only | 15 | 0.693 | 0.417 | 0.500 | 0.191 |

![特征消融](../../outputs/analysis/current/figures/feature_ablation_macro_f1.png)

消融结果表明，具体问卷答案是主要信号源。去除结果页描述符和互动元素后，macro-F1 从 `0.872` 降至 `0.787`。Answer-only 模型仍达到 `0.774` macro-F1，说明仅基于提交前问卷答案也能进行有效预测。

### 7.2 交叉验证稳定性

为避免单次 holdout 结果过于乐观，实验进一步进行了 5-fold cross-validation：

| 模型 | CV Macro-F1 Mean | CV Macro-F1 Std | Holdout Macro-F1 | Holdout Severe Error |
|---|---:|---:|---:|---:|
| LightGBM | 0.741 | 0.076 | 0.872 | 0.010 |
| XGBoost | 0.704 | 0.066 | 0.776 | 0.025 |
| Decision Tree | 0.716 | 0.069 | 0.735 | 0.035 |
| Random Forest | 0.699 | 0.026 | 0.718 | 0.030 |

![CV 稳定性](../../outputs/analysis/current/figures/cv_stability_macro_f1.png)

LightGBM 在 holdout 上表现最好，但 CV macro-F1 均值低于 holdout。这说明固定测试集结果可能偏乐观，少数类样本不足仍会带来性能波动。

### 7.3 有序分类与置信度分析

年龄分级天然具有顺序结构：

```text
3+ < 7+ < 12+ < 16+ < 18+
```

实验尝试将其转换为多个阈值二分类任务，但有序模型未超过直接多分类 LightGBM：

| 模型 | Accuracy | Macro-F1 | Severe Error |
|---|---:|---:|---:|
| Optimized LightGBM | 0.980 | 0.872 | 0.010 |
| Ordinal LightGBM Thresholds | 0.945 | 0.748 | 0.030 |

![有序分类对比](../../outputs/analysis/current/figures/ordinal_vs_multiclass_metrics.png)

此外，optimized LightGBM 的 Top-2 accuracy 达到 `0.995`，说明对于边界样本，真实标签通常仍位于模型前两个候选中。

![Top-2 置信度](../../outputs/analysis/current/figures/top2_confidence_bins.png)

### 7.4 区域评级扩展任务

除 IARC Generic 主标签外，结果页还包含多个地区评级机构的标签。实验进一步训练 LightGBM 预测不同地区评级，并在训练时移除 `result_age_rating`，避免标签泄漏。

| 机构 | 样本数 | 类别数 | Accuracy | Macro-F1 | Balanced Acc |
|---|---:|---:|---:|---:|---:|
| IARC Generic | 1322 | 5 | 0.975 | 0.829 | 0.767 |
| ESRB | 1322 | 5 | 0.884 | 0.822 | 0.807 |
| Google Play | 1322 | 6 | 0.960 | 0.811 | 0.787 |
| PEGI | 1322 | 6 | 0.960 | 0.748 | 0.744 |
| USK | 1322 | 5 | 0.970 | 0.738 | 0.749 |

![区域评级预测](../../outputs/analysis/current/figures/region_rating_prediction_performance.png)

该扩展实验表明，问卷答案不仅可以预测主标签，也包含迁移到不同地区评级体系的有效信号。

---

## 8. 讨论与总结

### 8.1 问题与解决方案

| 问题 | 表现 | 解决方案 |
|---|---|---|
| 问卷为树状结构 | 不同路径出现的问题不同 | 自动路径探索并记录可见问题 |
| 随机采样类别不均衡 | `18+` 占比过高 | 少数类补采与 macro-F1 评估 |
| 随机路径重复 | 采集到相同答案组合 | 使用 `response_signature` 去重 |
| 结果页结构复杂 | 多地区评级字段不统一 | 转换为标准 JSONL 和 CSV |
| 特征泄漏风险 | 结果页统计是后验信息 | 区分 full-feature 和 answer-only 实验 |
| 少数类指标不稳定 | `7+`、`16+` 测试样本少 | 使用 CV 和错误分析辅助解释 |

### 8.2 局限性

本实验仍存在若干限制：

1. 数据集类别不均衡，`3+`、`7+`、`16+` 样本仍然较少。
2. 模型是黑盒替代模型，只能近似输入输出行为，不能证明官方规则。
3. 部分统计特征来自结果页，严格提交前预测应优先参考 answer-only 结果。
4. 少数类 holdout support 较小，逐类别指标存在波动。
5. 采集范围受实验账号、应用类型和问卷版本影响，结论可能随平台更新变化。

### 8.3 结论

本文完成了 Google Play 年龄分级问卷的黑盒逆向建模实验。实验通过自动化路径探索采集了 `1322` 条真实有效样本，将树状问卷答案转换为 `1010` 维特征矩阵，并训练多种机器学习模型预测 IARC Generic 年龄分级。

优化后的 LightGBM 取得最佳综合表现，在 holdout test 上达到 `0.980` accuracy、`0.872` macro-F1 和 `0.010` severe error rate。特征消融表明，问卷答案是主要预测信号；错误分析显示模型主要产生保守高估，而没有严重低估 `18+` 样本；区域评级扩展进一步说明问卷特征对多个评级体系具有迁移价值。

总体而言，本实验表明：即使 Google Play 年龄分级规则不公开，也可以通过系统化采样和机器学习构建高性能替代模型。该模型不能取代官方评级机制，但能够帮助理解问卷答案与最终评级之间的关系，并为黑盒平台机制研究提供可复现的数据驱动方法。

---

## 附录 A：核心代码与脚本

### A.1 自动化采集模块

自动化采集模块负责从 Google Play 内容分级问卷中生成真实输入输出样本。该模块的核心不是简单提交固定表单，而是根据当前页面动态识别可见问题，并沿着条件分支逐步推进。

| 脚本 | 核心作用 | 输入 | 输出 |
|---|---|---|---|
| `scripts/probe_questionnaire_branches_cdp.py` | 探测问卷页面结构和可见问题 | 已登录浏览器页面 | 问卷问题和选项记录 |
| `scripts/sample_questionnaire_paths_cdp.py` | 随机探索问卷路径并采集样本 | 问卷页面、采样参数 | 原始路径样本 |
| `scripts/sample_minority_paths_cdp.py` | 针对少数类进行补充采样 | 低/中风险采样策略 | 少数类补充样本 |
| `scripts/convert_cdp_samples.py` | 转换采集结果为统一训练格式 | 原始 CDP 采样结果 | 标准 JSONL 样本 |

采集逻辑的核心循环如下：

```python
while questionnaire_can_continue(page):
    visible_questions = parse_visible_questions(page)
    question = choose_next_unanswered_question(visible_questions)
    option = sample_valid_option(question)
    select_option(page, question, option)
    answers[question.id] = option
    click_next_or_save(page)

rating_result = parse_rating_result(page)
```

为了避免重复采集同一答案组合，实验对每条样本生成答案签名：

```python
response_signature = hash_answer_combination(answers)

if response_signature in seen_signatures:
    skip_sample()
else:
    save_sample(answers, rating_result, response_signature)
```

这一机制使采集结果保留唯一问卷路径，有助于提高样本多样性。

### A.2 数据验证与特征构建模块

数据处理模块负责将原始采集结果转换为建模数据，并检查样本质量。

| 脚本 | 核心作用 | 关键检查或处理 |
|---|---|---|
| `scripts/validate_dataset.py` | 验证原始样本 | 检查有效样本、缺失标签、重复答案和状态分布 |
| `scripts/build_dataset.py` | 构建处理后数据集 | 展开问卷答案、保留标签、生成特征矩阵 |

数据读取中的关键细节如下：

```python
pd.read_csv(path, keep_default_na=False, na_values=[""])
```

该设置用于避免将问卷中的合法字符串 `"None"` 错误识别为缺失值。

特征构建时，问卷答案被展开为 one-hot 表示：

```python
answer_feature = f"answer__{question_id}_{selected_option}"
X[answer_feature] = 1
```

同时，系统保留路径统计和风险聚合特征，例如可见问题数量、跳过问题数量、内容描述符数量、互动元素数量以及各类风险主题得分。

### A.3 模型训练与优化模块

模型模块负责训练多分类年龄分级预测器，并保存评价结果。基础实验比较多个模型，优化实验进一步调整参数并以 macro-F1、balanced accuracy 和 severe error rate 作为主要选择依据。

| 脚本 | 核心作用 | 说明 |
|---|---|---|
| `scripts/train_models.py` | 训练基础模型 | 比较 baseline、线性模型、树模型和集成模型 |
| `scripts/optimize_models.py` | 模型调参与优化 | 搜索更优参数，选择 optimized LightGBM |

训练流程如下：

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = evaluate(y_test, pred)
```

年龄等级误差使用如下顺序：

```text
3+ -> 0
7+ -> 1
12+ -> 2
16+ -> 3
18+ -> 4
```

严重错误定义为预测等级与真实等级相差至少 2 级。该指标比普通 accuracy 更适合年龄分级任务，因为它能衡量跨年龄段的大幅误判。

### A.4 补充分析模块

补充分析模块用于提升实验解释性和说服力。

| 脚本 | 分析内容 | 实验目的 |
|---|---|---|
| `scripts/run_feature_ablation.py` | 特征消融 | 判断问卷答案、结果页统计、聚合特征的贡献 |
| `scripts/run_advanced_experiments.py` | 反事实、Top-2、有序分类 | 分析模型边界、置信度和标签顺序结构 |
| `scripts/train_region_rating_models.py` | 区域评级预测 | 测试问卷特征能否迁移到不同评级机构 |

这些分析使报告不仅给出最佳模型分数，还能回答模型为什么有效、哪些特征重要、结果是否稳定、是否存在后验特征影响等问题。

## 附录 B：主要输出文件

### B.1 数据产物

| 类型 | 文件 | 内容说明 | 用途 |
|---|---|---|---|
| 原始样本 | `data/raw/real_20260615_full.samples.jsonl` | 每行是一条真实问卷路径样本，包含答案、评级、描述符、互动元素和地区评级 | 保留最完整的实验原始证据 |
| 处理后数据 | `data/processed/real_20260615_full.dataset.csv` | 将原始样本整理为表格形式，保留标签和主要字段 | 用于数据统计和质量检查 |
| 特征矩阵 | `data/processed/real_20260615_full.features.csv` | 将问卷答案和派生特征转换为模型输入 | 用于模型训练、消融和扩展实验 |
| 问题目录 | `data/questionnaire/real_question_catalog_20260615_full.json` | 记录采集中发现的问题、选项和文本 | 用于解释特征含义和映射重要特征 |

这些数据产物形成从原始采集到建模输入的完整证据链。报告中的样本规模、标签分布和特征维度均由这些文件计算得到。

### B.2 指标与分析产物

| 类型 | 文件 | 内容说明 |
|---|---|---|
| 数据验证 | `outputs/analysis/current/metrics/dataset_validation_full.json` | 有效样本数、缺失标签数、重复样本数、标签分布等 |
| 基础模型指标 | `outputs/analysis/current/metrics/model_metrics.json` | 第一版多模型对比结果 |
| 优化模型指标 | `outputs/analysis/current/metrics/optimized_model_metrics.json` | 调参后模型性能和最佳模型结果 |
| 特征消融 | `outputs/analysis/current/metrics/feature_ablation_summary.csv` | 不同特征集合下的性能变化 |
| 交叉验证 | `outputs/analysis/current/metrics/optimized_cv_stability_summary.csv` | CV macro-F1、标准差和 holdout 对比 |
| 错误分析 | `outputs/analysis/current/metrics/optimized_error_analysis_summary.json` | 最佳模型错误数量、错误方向和严重错误情况 |
| 反事实分析 | `outputs/analysis/current/metrics/counterfactual_summary.json` | 高重要特征翻转后的预测变化 |
| 区域评级 | `outputs/analysis/current/metrics/region_rating_model_summary.csv` | 多个地区评级机构的预测性能 |

这些产物支撑报告中的主结果、稳定性分析、消融实验和扩展任务。它们保证报告中的结论不是主观描述，而是由可复查的实验输出支持。

### B.3 关键图表产物

| 图表 | 文件 | 报告中的作用 |
|---|---|---|
| 标签分布 | `outputs/analysis/current/figures/experiment_label_distribution.png` | 展示类别不均衡 |
| 基础模型对比 | `outputs/analysis/current/figures/model_performance.png` | 比较不同模型的初始表现 |
| 优化前后对比 | `outputs/analysis/current/figures/base_vs_optimized_macro_f1.png` | 展示调参收益 |
| 错误转移 | `outputs/analysis/current/figures/optimized_error_transitions.png` | 分析最佳模型错误方向 |
| 特征重要性 | `outputs/analysis/current/figures/lightgbm_feature_importance.png` | 解释关键影响特征 |
| 反事实转移 | `outputs/analysis/current/figures/counterfactual_prediction_transitions.png` | 展示局部特征翻转影响 |
| 特征消融 | `outputs/analysis/current/figures/feature_ablation_macro_f1.png` | 证明不同特征组贡献 |
| CV 稳定性 | `outputs/analysis/current/figures/cv_stability_macro_f1.png` | 展示交叉验证波动 |
| 区域评级扩展 | `outputs/analysis/current/figures/region_rating_prediction_performance.png` | 展示多地区评级预测能力 |

### B.4 过程截图

为避免附录堆叠过多页面截图，本文只选取三张能够串联采集流程的代表性截图：问卷填写、条件分支展开和结果页解析。截图仅用于说明实验过程，未展示账号密码、Cookie、浏览器登录态等敏感信息。

![图 B-1 问卷填写过程截图](../../outputs/analysis/current/screenshot/12.png)

图 B-1 展示了内容分级问卷的实际填写界面。页面中包含单选题、二选题和多项问答，自动化采集程序需要识别当前可见问题、读取候选选项，并记录本轮样本对应的答案组合。

![图 B-2 条件分支展开过程截图](../../outputs/analysis/current/screenshot/22.png)

图 B-2 展示了选择特定内容主题后继续展开的细分问题。该截图说明本实验面对的不是静态表格，而是具有条件跳转关系的树状问卷；前序答案会影响后续问题是否出现，因此采样策略必须覆盖不同路径，而不能只随机填写固定字段。

![图 B-3 评级结果页解析截图](../../outputs/analysis/current/screenshot/14.png)

图 B-3 展示了问卷提交后的评级结果页。结果页同时给出 IARC Generic 评级、地区评级、内容描述符和互动元素，本实验将这些内容解析为结构化标签与元信息，用于后续的数据验证、模型训练和区域评级扩展分析。
