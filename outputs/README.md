# outputs 目录说明

`outputs/` 现在按“原始采集”和“分析产物”分层管理，避免测试遗留物和正式结果混在一起。

## 目录约定

- `outputs/questionnaire_samples_cdp/`
  - 真实 Play Console 问卷采样结果。
  - `20260611_142334/` 是当前 1187 条有效候选样本对应的主采样批次，应作为数据事实源保留。

- `outputs/probes/`
  - 问卷结构探测、页面分支探测等中间产物。
  - `playwright/legacy/` 中保存的是旧的 `questionnaire_probe/` 历史结果。
  - `cdp/` 预留给新的 CDP 探测输出。

- `outputs/analysis/current/`
  - 当前默认分析输出目录。
  - 后续默认基于 1150 条最终建模样本运行验证、训练、解释和作图时，结果应写到这里。
  - 子目录：
    - `metrics/`
    - `models/`
    - `explanations/`
    - `figures/`

- `outputs/analysis/archive/`
  - 历史分析结果归档区。
  - `synthetic_baseline_30/` 保存的是此前基于 30 条模拟/测试数据生成的模型、指标和图表，不应作为最终实验结论使用。

## 使用建议

- 真实数据建模前，优先核对 `outputs/questionnaire_samples_cdp/20260611_142334/samples.jsonl`。
- 若要保留多轮分析结果，建议在 `outputs/analysis/archive/` 下按日期或数据版本新建子目录归档，再将新的正式结果写回 `outputs/analysis/current/`。
