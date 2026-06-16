# 数据采集阶段记录

## 概述

采集 Google Play Console IARC 内容分级问卷的随机路径样本，用于后续黑盒建模。

## 采集方案

采用 CDP (Chrome DevTools Protocol) + Playwright 远程连接方式，通过已登录的 Chrome 实例操作问卷页面。

### 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/probe_questionnaire_branches.py` | 问卷结构探测（提取问题、选项、跳转） |
| `scripts/probe_questionnaire_branches_cdp.py` | CDP 连接工具（页面选择、断点恢复） |
| `scripts/sample_questionnaire_paths_cdp.py` | 主采集脚本——随机路径探索 + 提交 + 结果解析 |

### 核心机制

- **随机路径探索**：每轮随机选择未回答问题的一个选项，模拟用户填写问卷的完整路径，直到出现 Save 按钮后提交
- **增量持久化**：每条样本采集完成后立即追加写入 `samples.jsonl`，崩溃不丢数据
- **断点续跑**：`--resume` 参数自动读取已有 JSONL，seed 自动偏移避免重复路径
- **Chrome 崩溃恢复**：检测 `TargetClosedError` 和 `chrome-error` 后自动重连 CDP
- **去重保护**：基于 `response_signature`（SHA1-48bit 哈希）过滤重复答案组合

### 采样参数

```text
--sample-count 1000
--seed 20260611     (后续 resume 时自动偏移为 20261143)
--settle-ms 900
--page-index 0
```

## 数据集

### 基础采样批次

初始大规模 CDP 采样文件为 `outputs/questionnaire_samples_cdp/20260611_142334/samples.jsonl`。

| 指标 | 数值 |
|------|------|
| 总采集数 | 1187 |
| 有效样本（complete） | 1187 |
| 完成率 | 100.0% |
| 唯一签名数 | 1187（零重复） |

该基础池评级分布如下：

| 评级 | 数量 | 占比 |
|------|------|------|
| Rated for 18+ | 989 | 83.3% |
| Rated for 12+ | 143 | 12.0% |
| Rated for 3+ | 26 | 2.2% |
| Rated for 16+ | 26 | 2.2% |
| Rated for 7+ | 3 | 0.3% |

### 少数类补采后全量口径

后续补采批次 `minority_supplement_20260614_204627`、`minority_supplement_20260614_retry1`、`minority_supplement_20260614_retry2`、`minority_supplement_20260615_formal_low_v2`、`minority_supplement_20260615_formal_low_v4`、`minority_supplement_20260615_formal_low_v5_100` 已并入当前全量训练池。

当前主样本文件为 `data/raw/real_20260615_full.samples.jsonl`，对应转换报告为 `outputs/analysis/current/metrics/conversion_report_full.json`。

| 指标 | 数值 |
|------|------|
| 输入记录数 | 1341 |
| 完成记录数 | 1339 |
| 循环终止（loop_detected） | 2 |
| 去重后有效样本 | 1322 |
| 重复签名跳过 | 17 |
| 问题总数 | 229 |

当前全量训练池评级分布如下：

| 评级 | 数量 | 占比 |
|------|------|------|
| Rated for 18+ | 1013 | 76.6% |
| Rated for 12+ | 225 | 17.0% |
| Rated for 3+ | 37 | 2.8% |
| Rated for 16+ | 28 | 2.1% |
| Rated for 7+ | 19 | 1.4% |

其中低龄样本 `3+ / 7+ / 12+` 合计 `281` 条，占 `21.3%`，较基础池的 `176` 条有明显提升。

### 历史子集说明

早期为了控制实验规模，曾使用固定随机种子 `20260613` 从基础池中抽取 `1150` 条作为建模子集；该文件仍保留在 `data/raw/real_20260611_142334_n1150.samples.jsonl`，仅用于和旧版实验结果对照，不再作为当前默认训练集。

### 失败样本分析

当前并入训练池的 `1341` 条输入记录中，仅有 `2` 条为 `loop_detected`，其余完成记录均成功解析并完成去重，没有保留 `rating_extraction_failed` 类型样本。

历史试跑过程中确实出现过 `Next did not become enabled after Save`，本质上是 Google 服务端在 Save 完成后延迟亮起 Continue 按钮导致的偶发等待超时。该问题属于服务端波动，未进入当前全量训练池。

### 每条样本包含字段

```text
sample_id, status, error, path, path_id, path_summary,
responses, response_signature, answer_count, continue_count,
final_url, final_title, final_question_count,
final_can_continue, final_can_finalize, final_errors,
final_state_signature, rating_result, rating_signature,
completed_at
```

其中 `rating_result` 包含：
```text
ok, error, ratings, primary_authority, primary_rating,
primary_content_descriptors, primary_interactive_elements,
summary_url, summary_title, summary_state_signature,
summary_body_fingerprint, summary_body_excerpt
```

## 遇到的问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `TargetClosedError` | Chrome tab 或浏览器进程崩溃 | 自动重连 CDP + 页面恢复 |
| `chrome-error://chromewebdata/` | 浏览器内存不足 / 渲染进程崩溃 | 检测后自动 cooldown + 重试 |
| `ERR_CONNECTION_CLOSED` | CDP WebSocket 断连 | 重新获取 `webSocketDebuggerUrl` 并重连 |
| 连续大量 SKIP duplicate | resume 时相同 seed 重放旧路径 | seed 自动偏移 = 原始 seed + 已有样本数 |
| 进度显示分母变大 | 全局编号 + 已有样本 = 混淆的 `total_target` | 改为显示本次进度/本次目标 |
| `Next did not become enabled after Save` | Google 服务端延迟 | 作为历史偶发错误记录，当前全量训练池中无此类保留样本 |
