# play-rating-blackbox

Google Play 年龄分级问卷黑盒逆向建模实验代码框架。

## 环境准备

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
```

可选加速和解释包：

```bash
.venv/bin/pip install -r requirements-optional.txt
```

## 本地模拟链路

无 Play Console 访问时，可以先用模拟问卷跑通完整链路：

```bash
.venv/bin/python scripts/map_questionnaire.py
.venv/bin/python scripts/generate_synthetic_samples.py --count 30 --strategy all
.venv/bin/python scripts/validate_dataset.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/train_models.py
.venv/bin/python scripts/explain_models.py
.venv/bin/python scripts/make_figures.py
```

主要输出：

- `data/raw/samples.jsonl`
- `data/processed/dataset.csv`
- `outputs/analysis/current/metrics/model_metrics.json`
- `outputs/analysis/current/explanations/`
- `outputs/analysis/current/figures/`

## 真实采集

先安装 Playwright 浏览器，并确认 `configs/collector.yaml` 和 `.env.local`/`docs/key.md` 中有目标 Console 信息。

首次建议先 dry-run：

```bash
.venv/bin/python scripts/collect_samples.py --limit 1 --strategy baseline --dry-run
```

确认页面识别和证据保存正常后，再运行真实提交：

```bash
.venv/bin/python scripts/collect_samples.py --limit 10 --strategy baseline --resume
```

脚本遇到登录失效、验证码、安全检查、政策警告或结果解析失败时会停止，并追加记录到 `docs/人工操作清单.md`。

## 真实数据对接

CDP 真实采样结果需要先转换成训练流水线使用的标准 JSONL：

```bash
.venv/bin/python scripts/convert_cdp_samples.py --input \
  outputs/questionnaire_samples_cdp/20260611_142334/samples.jsonl \
  outputs/questionnaire_samples_cdp/minority_supplement_20260614_204627/samples.jsonl \
  outputs/questionnaire_samples_cdp/minority_supplement_20260614_retry1/samples.jsonl \
  outputs/questionnaire_samples_cdp/minority_supplement_20260614_retry2/samples.jsonl \
  outputs/questionnaire_samples_cdp/minority_supplement_20260615_formal_low_v2/samples.jsonl \
  outputs/questionnaire_samples_cdp/minority_supplement_20260615_formal_low_v4/samples.jsonl \
  outputs/questionnaire_samples_cdp/minority_supplement_20260615_formal_low_v5_100/samples.jsonl \
  --output data/raw/real_20260615_full.samples.jsonl \
  --report outputs/analysis/current/metrics/conversion_report_full.json \
  --catalog data/questionnaire/real_question_catalog_20260615_full.json
.venv/bin/python scripts/validate_dataset.py --input data/raw/real_20260615_full.samples.jsonl --output outputs/analysis/current/metrics/dataset_validation_full.json
.venv/bin/python scripts/build_dataset.py --input data/raw/real_20260615_full.samples.jsonl --dataset-output data/processed/real_20260615_full.dataset.csv --features-output data/processed/real_20260615_full.features.csv
```

当前全量真实数据集为 `data/raw/real_20260615_full.samples.jsonl`，包含去重后的 `1322` 条有效样本，评级分布为：

- `3+`: `37`
- `7+`: `19`
- `12+`: `225`
- `16+`: `28`
- `18+`: `1013`

转换报告写入 `outputs/analysis/current/metrics/conversion_report_full.json`，问题字典写入 `data/questionnaire/real_question_catalog_20260615_full.json`，校验报告写入 `outputs/analysis/current/metrics/dataset_validation_full.json`。
历史上的 `1150` 条子集仍保留在 `data/raw/real_20260611_142334_n1150.samples.jsonl`，仅用于与早期实验结果对比。

## 少数类补采

如果需要改善年龄分级分布，可以使用少数类导向采样脚本。该脚本复用 CDP 提交和结果解析逻辑，但将答案选择策略从纯随机改为低/中风险导向，并默认排除已采集样本的 `response_signature`。
默认还会按风险档位过滤目标评级：`low` 只计入 `3+ / 7+ / 12+`，`medium` 只计入 `7+ / 12+ / 16+`，`mixed` 只计入 `3+ / 7+ / 12+ / 16+`。如果某次运行里出现大量 `18+`，它们会被直接跳过，不会算进完成数。

```bash
.venv/bin/python scripts/sample_minority_paths_cdp.py --endpoint-url http://127.0.0.1:9222 --risk-profile low --sample-count 50 --assume-ready
.venv/bin/python scripts/sample_minority_paths_cdp.py --endpoint-url http://127.0.0.1:9222 --risk-profile medium --sample-count 100 --assume-ready
```

如果采集中途停止，可以在相同的 `--output-dir` 下追加 `--resume` 继续；此时 `--sample-count` 表示该目录的总目标样本数，而不是“再新增多少条”。

推荐用途：

- `low`：尽量补 `3+`、`7+` 附近样本。
- `medium`：尽量补 `7+`、`12+`、`16+` 边界样本。
- `mixed`：默认策略，在低/中风险之间随机切换。

补采结果会写入 `outputs/questionnaire_samples_cdp/minority_supplement_*`，建议先单独验证分布，再决定是否合并进最终训练集。

## 安全边界

- `docs/key.md`、`.env.local`、浏览器 profile、Cookie、原始截图和 HTML 默认不进入 Git。
- 代码可以读取 `docs/key.md`，但不会打印或写出账号密码。
- 报告中如需页面证据，应先脱敏后再放入可提交目录。
