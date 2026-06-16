# Google Play 年龄分级问卷黑盒逆向建模实验设计与执行计划

## 0. 实验定位

本实验不是简单地“随机填问卷 + 跑模型”，而是要把 Google Play Content Rating Questionnaire 视为一个可查询但规则不公开的黑盒函数：

```text
问卷答案向量 x  ->  Google Play / IARC 分级结果 y
```

目标是通过系统采样、自动化提交、机器学习建模和可解释性分析，尽可能还原问卷答案与最终年龄分级之间的映射关系，并形成一份有实验设计亮点、数据可信、分析充分、图表完整的课程报告。

本实验应突出三个亮点：

1. **树结构感知的数据采集**：不是盲目随机采样，而是先重建问卷的条件跳转结构，再进行覆盖导向采样、边界采样和主动学习采样。
2. **多层次预测任务**：不仅预测最终年龄分级，还尽量记录内容描述符、互动元素、地区分级差异，形成更丰富的黑盒建模对象。
3. **可解释性反推机制**：使用决策树、特征重要性、SHAP/Permutation Importance、局部反事实样本等方法分析哪些问卷答案最影响分级。

## 1. 官方背景与实验边界

根据 Google Play 官方 Help Center，应用内容分级由多个独立评级机构根据开发者在 Play Console 中填写的内容分级问卷回答生成；不同地区可能展示不同评级结果。官方政策也强调，错误陈述应用内容可能导致应用被移除或暂停。因此实验开始时仍应复查最新文档和 Console 实际页面，确认入口、字段名称和评级展示方式没有变化。

因此本实验必须遵守以下边界：

- 只使用自己有权限访问的 Google Play Developer Console 账号和测试应用。
- 不绕过登录、MFA、验证码、风控或任何访问控制。
- 不发布误导性应用，不向真实用户分发实验应用。
- 自动化只用于重复填写和读取自己账号下的问卷结果；如果 Play Console 出现风控、验证码或限制，应暂停自动化并改用人工确认。
- 不保存账号密码、Cookie、个人隐私数据或支付信息到仓库。
- 控制请求频率，保留人工检查点，避免对 Console 服务造成异常压力。

已核验官方资料：

- Google Play Console Help: Content Ratings  
  https://support.google.com/googleplay/android-developer/answer/9898843
- Google Play Console Help: Requirements related to content ratings for apps, games and ads  
  https://support.google.com/googleplay/android-developer/answer/9859655
- IARC official site  
  https://www.globalratings.com/

## 2. 总体执行路线

实验分为 7 个阶段：

1. **问卷结构探索**：人工进入 Play Console，记录问卷页面、问题文本、选项、条件跳转和结果页面字段。
2. **自动化框架搭建**：用 Playwright/Selenium 实现半自动问卷填写、提交、截图、结果解析和日志记录。
3. **小规模试采集**：采集 30-50 条样本，验证自动化稳定性、字段解析正确性和结果可复现性。
4. **正式采样设计**：结合树结构覆盖、随机采样、分层采样、边界采样和主动学习采样，收集至少 1000 条有效样本。
5. **数据清洗与特征工程**：将问卷答案、派生特征、分级结果、内容描述符、互动元素整理成可训练数据集。
6. **模型训练与评估**：训练至少 3 类模型，比较多分类预测性能和类别差异。
7. **机制分析与报告写作**：从特征重要性、规则提取、反事实样本和错误案例中总结 Google Play 分级机制的潜在模式。

## 3. 数据采集设计

### 3.1 需要采集的原始信息

每条样本至少保存以下字段：

```text
sample_id
timestamp
app_id / test_app_name
questionnaire_version
category_path
answers_json
visible_questions
skipped_questions
submit_status
result_age_rating
result_region_ratings
content_descriptors
interactive_elements
certificate_or_result_id
page_screenshot_path
html_snapshot_path
automation_log_path
notes
```

其中 `answers_json` 应保留完整问卷答案，不要只保存 one-hot 后的特征。原始答案是后续复查、修复编码和写报告的关键证据。

### 3.2 问卷结构建模

由于问卷是树状/条件跳转结构，第一步不是采样，而是建立问卷结构图：

```text
question_id
question_text
answer_options
parent_question_id
condition_to_show
children_question_ids
question_group
is_terminal_related
```

建议输出两个文件：

```text
data/questionnaire/question_schema.json
data/questionnaire/question_tree.graphml
```

报告中可将问卷结构图作为亮点图表：展示不同内容类别如何触发后续问题，例如暴力、性内容、语言、药物、赌博、用户生成内容、位置分享、在线互动等。

### 3.3 采样策略

正式数据集建议由 5 类样本组成，而不是全部随机生成：

| 样本类型 | 数量建议 | 目的 |
|---|---:|---|
| 全否定/低风险基线样本 | 30 | 获取最低分级与默认描述符 |
| 单因素扰动样本 | 200 | 分析每个问题单独打开时的影响 |
| 树结构覆盖样本 | 250 | 保证每个主要分支和叶子问题都被覆盖 |
| 分层随机样本 | 350 | 提供训练模型所需的自然多样性 |
| 边界/主动学习样本 | 200 | 强化模型容易混淆的 7+/12+/16+/18+ 边界 |

总数约 1030 条，留出无效样本空间，确保最终有效样本不少于 1000。

### 3.4 样本生成方法

采用分阶段生成：

#### 阶段 A：基线样本

构造所有敏感内容均为否定的样本，验证最低年龄分级。对互动元素分别打开/关闭，例如：

- 是否允许用户互动
- 是否共享位置
- 是否包含数字购买
- 是否访问互联网

这些通常不一定改变年龄分级，但会影响 Interactive Elements，是报告中的重要分析点。

#### 阶段 B：单因素扰动

以低风险基线为起点，每次只改变一个问题或一个小问题簇：

```text
baseline + mild violence
baseline + realistic violence
baseline + profanity
baseline + simulated gambling
baseline + sexual content
baseline + drug reference
baseline + user-generated content
```

目的：估计每个问题对最终分级的边际影响。

#### 阶段 C：路径覆盖采样

遍历问卷树，使每个可见问题、关键选项和叶子路径至少出现若干次。可以把目标定义为：

```text
question_coverage >= 95%
option_coverage >= 90%
leaf_path_coverage >= 80%
```

如果问卷分支很多，叶子路径不必全覆盖，但要覆盖所有高风险内容类别和常见中风险组合。

#### 阶段 D：分层随机采样

按风险主题分层抽样，避免数据集中低风险样本过多：

```text
violence: 20%
sexual_content: 15%
language: 15%
drugs: 10%
gambling: 10%
fear: 10%
ugc_interaction: 10%
mixed_high_risk: 10%
```

每个主题内部再随机选择严重程度、出现频率、是否真实/幻想、是否图像化等选项。

#### 阶段 E：主动学习与边界采样

先用前 600-700 条样本训练初版模型，找出模型不确定性最高或类别边界最模糊的区域，再生成追加样本：

```text
P(12+) ≈ P(16+)
P(16+) ≈ P(18+)
模型之间预测不一致
决策树边界附近
单个问题改变导致分级跳变
```

这个阶段是报告亮点：说明实验不是机械采集，而是根据模型反馈优化采样。

## 4. 自动化实现方案

### 4.1 推荐技术栈

```text
Python 3.11+
Playwright
pandas
scikit-learn
xgboost 或 lightgbm
matplotlib / seaborn / plotly
shap
SQLite + JSONL
```

Playwright 比 Selenium 更适合现代前端页面，选择器、等待机制和截图功能更稳定。

### 4.2 自动化模块拆分

建议代码结构：

```text
src/
  collector/
    browser_session.py        # 浏览器启动、登录态复用、限速
    questionnaire_mapper.py   # 识别问题、选项、跳转结构
    sample_generator.py       # 生成待提交答案
    submitter.py              # 自动填写和提交
    result_parser.py          # 解析结果页
    validator.py              # 样本有效性检查
  data/
    schema.py                 # 数据结构定义
    storage.py                # SQLite / JSONL 写入
  modeling/
    features.py               # 特征工程
    train.py                  # 训练入口
    evaluate.py               # 指标计算
    explain.py                # 可解释性分析
  visualization/
    plots.py                  # 图表生成
scripts/
  map_questionnaire.py
  collect_samples.py
  validate_dataset.py
  train_models.py
  make_figures.py
```

### 4.3 自动化工作流

自动化不应直接从登录开始硬编码账号密码。推荐流程：

1. 人工打开 Playwright 持久化浏览器。
2. 人工登录 Google Play Console。
3. 脚本复用本地浏览器 profile。
4. 脚本进入测试应用的内容分级页面。
5. 按样本计划填写问卷。
6. 提交前保存答案快照。
7. 提交后解析结果页。
8. 保存截图、HTML 快照、结构化结果。
9. 如果页面异常、验证码、登录过期或结果无法解析，立即停止并等待人工处理。

### 4.4 速率和稳定性控制

建议配置：

```text
min_delay_between_submissions = 20-60 seconds
max_samples_per_batch = 100
manual_review_every_n_samples = 50
retry_per_sample = 1
stop_on_captcha = true
stop_on_policy_warning = true
```

每批采集后运行一次验证：

```bash
python scripts/validate_dataset.py --input data/raw/samples.jsonl
```

验证内容：

- 样本是否有完整答案。
- 是否有最终年龄分级。
- 是否有截图和 HTML 证据。
- 是否出现重复样本。
- 是否结果类别过度单一。
- 是否有解析失败或提交失败。

## 5. 数据质量控制

### 5.1 有效样本定义

一条样本只有同时满足以下条件才计为有效：

- 问卷成功提交。
- 结果页成功展示并被保存。
- 至少包含一个目标分级字段。
- 原始答案完整保存。
- 没有自动化异常、页面解析异常或人工中断标记。
- 与已有样本不是完全重复答案。

### 5.2 多样性指标

采集完成后要报告以下数据集指标：

```text
样本总数
有效样本数
无效样本数
最终年龄分级分布
每个问题的选项覆盖率
每个主题分支的覆盖率
平均可见问题数量
平均跳过问题数量
唯一答案组合数量
重复率
```

### 5.3 类别不平衡处理

年龄分级通常可能不均衡，尤其是低风险类别更容易出现。应使用：

- 分层采样补足少数类。
- 训练时使用 `class_weight=balanced`。
- 报告 macro-F1、balanced accuracy，而不是只报告 accuracy。
- 对少数类单独分析召回率。

## 6. 特征工程

### 6.1 基础特征

将问卷答案编码为：

- 二元问题：`0/1`
- 单选题：one-hot
- 多选题：multi-hot
- 有序严重程度：ordinal encoding
- 未出现的问题：单独编码为 `not_visible`，不要简单当作 `No`

`not_visible` 很重要，因为“问题没出现”和“问题出现但回答否定”在树结构问卷中语义不同。

### 6.2 主题聚合特征

额外构造主题级特征：

```text
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
visible_question_count
triggered_branch_count
```

这些特征有两个价值：

1. 提升模型泛化。
2. 在报告中更容易解释分级机制。

### 6.3 交互特征

加入关键组合特征：

```text
violence + realistic
violence + blood
sexual_content + nudity
gambling + real_money
ugc + user_interaction
internet_access + user_generated_content
```

黑盒规则通常存在阈值和组合效应，交互特征能帮助线性模型捕捉非线性边界。

## 7. 建模方案

### 7.1 预测目标

主任务：

```text
预测 Google Play / IARC Generic 年龄分级：3+ / 7+ / 12+ / 16+ / 18+
```

扩展任务：

```text
预测地区分级：ESRB / PEGI / USK / ClassInd / ACB 等
预测 Content Descriptors
预测 Interactive Elements
```

如果时间有限，主任务必须完成；扩展任务可以作为报告亮点。

### 7.2 模型列表

至少训练以下 5 类模型，报告中重点比较其中 3-4 个：

| 模型 | 作用 | 预期特点 |
|---|---|---|
| Majority / Stratified Baseline | 下限基线 | 证明任务不是靠类别分布取巧 |
| Logistic Regression | 可解释线性模型 | 提供系数方向和基础可解释性 |
| Decision Tree | 规则近似模型 | 可视化黑盒规则边界 |
| Random Forest / Extra Trees | 稳健树模型 | 通常性能强，能输出特征重要性 |
| XGBoost / LightGBM | 高性能模型 | 适合表格数据和非线性组合 |

如时间允许可加入：

- SVM
- MLP
- CatBoost

### 7.3 训练与调参

使用分层划分：

```text
train: 70%
validation: 15%
test: 15%
```

或使用：

```text
Stratified 5-fold cross validation + final holdout test
```

调参范围示例：

```text
DecisionTree:
  max_depth: [3, 5, 8, 12, None]
  min_samples_leaf: [1, 5, 10, 20]

RandomForest:
  n_estimators: [200, 500]
  max_depth: [8, 12, None]
  min_samples_leaf: [1, 3, 5]
  class_weight: [balanced]

XGBoost/LightGBM:
  n_estimators: [200, 500]
  max_depth: [3, 5, 8]
  learning_rate: [0.03, 0.05, 0.1]
  subsample: [0.8, 1.0]
```

## 8. 评估指标

报告中至少包含：

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1
Balanced Accuracy
Confusion Matrix
Per-class Precision / Recall / F1
```

如果模型输出概率，再加入：

```text
Top-2 Accuracy
Calibration Curve
Brier Score
```

年龄分级是有序类别，因此建议额外报告：

```text
Mean Absolute Age-Level Error
Severe Error Rate
```

其中：

```text
3+ -> 0
7+ -> 1
12+ -> 2
16+ -> 3
18+ -> 4
```

`Severe Error` 定义为预测等级与真实等级相差大于等于 2，例如真实 18+ 预测为 12+ 或更低。

## 9. 结果分析设计

### 9.1 整体性能分析

回答：

- 哪个模型总体效果最好？
- 复杂模型相比线性模型提升多少？
- 模型是否明显优于基线？
- 错误主要集中在哪些类别之间？

### 9.2 类别差异分析

重点分析：

- `3+` 是否最容易预测？
- `12+` 和 `16+` 是否容易混淆？
- `18+` 是否由少数高风险问题强触发？
- 少数类别是否因为样本不足而召回率低？

### 9.3 特征重要性分析

使用多种方法交叉验证：

```text
Decision Tree split rules
Random Forest feature importance
Permutation Importance
SHAP values
单因素扰动结果
```

报告中不要只给一个重要性排名，应解释“为什么这些问题重要”，例如：

- 暴力内容的真实程度、血腥程度、频率可能显著影响分级。
- 性内容、赌博、毒品相关问题可能形成高等级触发器。
- 互动元素可能更多影响描述符，而不一定直接提升年龄分级。

### 9.4 反事实样本分析

构造局部反事实：

```text
样本 A: 12+
仅将 simulated gambling 从 No 改为 Yes -> 16+

样本 B: 16+
仅将 sexual nudity 从 mild 改为 explicit -> 18+
```

这种分析非常适合写进报告，因为它比单纯模型准确率更像“逆向规则发现”。

### 9.5 错误案例分析

从测试集挑选 10-20 个错误样本，按错误类型分类：

```text
边界模糊
少数类样本不足
多个中风险因素叠加
地区分级规则差异
问题未出现导致特征缺失
```

每类给一个具体样本，说明模型为什么可能预测错。

## 10. 图表清单

报告中建议至少包含 10 张高质量图表：

| 图表 | 用途 |
|---|---|
| 实验流程图 | 展示采集、训练、分析全流程 |
| 问卷树结构图 | 展示条件跳转和主题分支 |
| 样本来源组成图 | 展示不同采样策略占比 |
| 年龄分级分布柱状图 | 展示标签分布 |
| 问题/选项覆盖率热力图 | 证明数据多样性 |
| 主题风险分布图 | 展示 violence/sex/gambling 等主题覆盖 |
| 模型性能对比表/柱状图 | 比较 3+ 模型 |
| 混淆矩阵 | 分析类别混淆 |
| 特征重要性 Top 20 | 解释关键问题 |
| SHAP summary plot | 解释非线性模型 |
| 单因素扰动影响图 | 展示问题对分级的边际影响 |
| 反事实样本表 | 展示分级跳变边界 |

## 11. 报告结构建议

最终报告建议采用论文式结构：

```text
1. Introduction
   1.1 Background
   1.2 Research Question
   1.3 Contributions

2. Google Play Age Rating Questionnaire
   2.1 Content Rating Workflow
   2.2 Black-box Modeling Setting
   2.3 Ethical and Practical Constraints

3. Data Collection
   3.1 Questionnaire Structure Reconstruction
   3.2 Sampling Strategy
   3.3 Automation Implementation
   3.4 Dataset Quality Control
   3.5 Dataset Statistics

4. Feature Engineering
   4.1 Raw Answer Encoding
   4.2 Tree-aware Missing Value Encoding
   4.3 Risk Theme Aggregation
   4.4 Interaction Features

5. Model Training
   5.1 Task Definition
   5.2 Models
   5.3 Training Protocol
   5.4 Metrics

6. Results
   6.1 Overall Performance
   6.2 Per-class Performance
   6.3 Confusion Matrix Analysis
   6.4 Ablation Study

7. Interpreting the Black Box
   7.1 Feature Importance
   7.2 Decision Rules
   7.3 Counterfactual Analysis
   7.4 Regional Rating Differences

8. Problems and Solutions

9. Conclusion

Appendix
   Core code snippets
   Dataset schema
   Additional figures
```

## 12. 仓库文件规划

建议最终形成如下目录：

```text
docs/
  todo.md
  experiment_notes.md
  report_outline.md
data/
  questionnaire/
    question_schema.json
    question_tree.graphml
  raw/
    samples.jsonl
    screenshots/
    html/
  processed/
    dataset.csv
    features.parquet
    label_mapping.json
  splits/
    train.csv
    valid.csv
    test.csv
outputs/
  analysis/
    current/
      figures/
      models/
      metrics/
      explanations/
    archive/
  probes/
  questionnaire_samples_cdp/
src/
  collector/
  data/
  modeling/
  visualization/
scripts/
  map_questionnaire.py
  collect_samples.py
  validate_dataset.py
  train_models.py
  make_figures.py
report/
  report.md
  report.pdf
```

`.gitignore` 中应忽略：

```text
.env
.env.local
browser_profile/
data/raw/screenshots/
data/raw/html/
*.sqlite
*.db
```

如果截图或 HTML 需要作为报告证据，可以挑选少量脱敏样例放入 `report/assets/`。

## 13. 执行时间表

### Day 1：准备与问卷理解

- [ ] 阅读官方文档和实验要求。
- [ ] 建立测试应用，不发布到真实用户。
- [ ] 人工完成 3-5 次问卷，观察页面结构和结果字段。
- [ ] 初步记录问题、选项、条件跳转。
- [ ] 确定主标签：优先使用 IARC Generic 或 Google Play 展示分级。

### Day 2：自动化骨架

- [ ] 搭建 Python + Playwright 环境。
- [ ] 实现持久化浏览器 profile。
- [ ] 实现问题识别、选项点击、提交前截图。
- [ ] 实现结果页截图和字段解析。
- [ ] 保存 JSONL 原始样本。

### Day 3：小规模试采集

- [ ] 采集 30-50 条样本。
- [ ] 检查结果是否可复现。
- [ ] 修正选择器和解析逻辑。
- [ ] 建立样本有效性验证脚本。
- [ ] 输出初版 question_schema.json。

### Day 4-5：正式采集第一轮

- [ ] 采集基线样本。
- [ ] 采集单因素扰动样本。
- [ ] 采集树路径覆盖样本。
- [ ] 每 50-100 条做一次人工抽检。
- [ ] 生成中期数据质量报告。

### Day 6：初版建模与主动学习

- [ ] 生成 processed dataset。
- [ ] 训练 baseline、logistic regression、decision tree、random forest。
- [ ] 找出错误样本和低置信度区域。
- [ ] 设计追加样本，重点补齐边界类别。

### Day 7：正式采集第二轮

- [ ] 采集主动学习样本。
- [ ] 确保有效样本数不少于 1000。
- [ ] 检查类别分布和选项覆盖率。
- [ ] 冻结数据集版本 `dataset_v1`。

### Day 8：最终模型与解释性分析

- [ ] 训练最终模型。
- [ ] 完成交叉验证和 holdout test。
- [ ] 生成混淆矩阵、性能对比图。
- [ ] 计算特征重要性和 SHAP。
- [ ] 整理反事实样本。

### Day 9-10：报告写作与打磨

- [ ] 写完整实验报告。
- [ ] 加入核心代码片段。
- [ ] 加入所有关键图表。
- [ ] 写清楚问题与解决方案。
- [ ] 检查结论是否由数据支持。
- [ ] 最终导出 PDF。

## 14. 最小可行版本与高质量版本

### 最小可行版本

必须完成：

- 1000 条有效样本。
- 3 个模型。
- 基础特征工程。
- accuracy、macro-F1、混淆矩阵。
- 数据采集策略说明。
- 完整报告。

### 高质量版本

争取完成：

- 问卷树结构图。
- 主动学习采样。
- 单因素扰动实验。
- SHAP 和 permutation importance。
- 反事实分级跳变分析。
- 地区分级差异分析。
- 自动化稳定性和数据质量验证脚本。
- 图文并茂、接近小论文质量的报告。

## 15. 风险与备选方案

| 风险 | 表现 | 备选方案 |
|---|---|---|
| Play Console 限制自动化 | 登录失效、验证码、风控提示 | 降低频率，使用人工确认，减少批量规模 |
| 问卷结构变化 | 选择器失效、问题文本变化 | 使用文本匹配 + 结构快照，记录 questionnaire_version |
| 结果类别不均衡 | 模型只会预测多数类 | 分层补采少数类，使用 class_weight 和 macro-F1 |
| 某些高风险组合触发政策警告 | 无法继续提交 | 停止该类组合，记录为不可采区域，不绕过限制 |
| 样本重复过多 | 有效样本不足 | 用答案哈希去重，增加路径覆盖和主动学习样本 |
| 页面结果解析不稳定 | 字段缺失 | 保存截图和 HTML，允许人工回填少量样本 |

## 16. 当前下一步清单

近期优先执行：

- [ ] 创建项目目录结构。
- [ ] 建立 Python 环境和依赖文件。
- [ ] 人工进入 Play Console，确认内容分级问卷入口和结果页字段。
- [ ] 人工完成 3 个样例，记录问题结构。
- [ ] 编写 `question_schema.json` 初版。
- [ ] 实现 Playwright 持久化浏览器启动脚本。
- [ ] 实现单条样本自动填写与结果保存。
- [ ] 跑 30 条 pilot 数据，确认流程可行后再进入 1000 条采集。

## 17. 报告中的核心论点预设

最终报告可以围绕以下论点展开，但必须等真实数据验证后再写成结论：

1. Google Play 年龄分级虽然是黑盒，但问卷答案与分级之间存在较强的可学习映射。
2. 低风险和极高风险类别更容易预测，中间年龄段更容易混淆。
3. 少数高风险问题可能具有强触发器作用，而多个中风险问题可能存在叠加效应。
4. 互动元素更可能影响描述符，而不是直接决定年龄分级。
5. 树结构感知采样比纯随机采样更能提高类别覆盖和边界样本质量。
