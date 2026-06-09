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
- `outputs/metrics/model_metrics.json`
- `outputs/explanations/`
- `outputs/figures/`

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

## 安全边界

- `docs/key.md`、`.env.local`、浏览器 profile、Cookie、原始截图和 HTML 默认不进入 Git。
- 代码可以读取 `docs/key.md`，但不会打印或写出账号密码。
- 报告中如需页面证据，应先脱敏后再放入可提交目录。
