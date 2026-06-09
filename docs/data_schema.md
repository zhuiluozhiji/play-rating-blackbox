# 数据结构说明

## 原始样本 `data/raw/samples.jsonl`

每行是一条 JSON 样本，核心字段如下：

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本唯一 ID |
| `timestamp` | 采集时间 |
| `strategy` | 样本生成策略 |
| `questionnaire_version` | 问卷版本或页面指纹 |
| `answers_json` | 原始问卷答案 |
| `visible_questions` | 采集时可见问题 ID |
| `skipped_questions` | 由条件跳转导致未出现的问题 ID |
| `result_age_rating` | 主预测标签 |
| `result_region_ratings` | 不同地区/评级机构结果 |
| `content_descriptors` | 内容描述符 |
| `interactive_elements` | 互动元素 |
| `status` | `success` 或失败状态 |
| `evidence` | 截图、HTML、日志等本地证据路径 |

## 处理后数据 `data/processed/dataset.csv`

处理后数据保留标签字段，并展开问卷答案特征。树结构问卷中未出现的问题编码为 `not_visible`，不会直接等同于 `No`。

## 模型输出

- `outputs/metrics/model_metrics.json`：各模型整体指标。
- `outputs/metrics/per_class_metrics.csv`：各类别 precision/recall/F1。
- `outputs/metrics/confusion_matrix_*.csv`：混淆矩阵。
- `outputs/models/*.joblib`：训练后的模型。
- `outputs/figures/*.png`：报告图表。
