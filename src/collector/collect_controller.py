from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, List, Optional

from src.collector.browser_session import BrowserConfig, launch_persistent_context
from src.collector.key_reader import read_credentials
from src.collector.manual_ops import append_manual_action
from src.collector.sample_generator import generate_samples
from src.collector.submitter import submit_sample
from src.common import load_yaml, read_json
from src.data.schema import default_question_schema
from src.data.storage import JsonlStore


def load_schema(path: str) -> Dict[str, Any]:
    schema = read_json(path, default=None)
    return schema or default_question_schema()


async def collect_samples_async(
    config_path: str,
    limit: int,
    strategy: str,
    start_url: Optional[str],
    dry_run: bool,
    no_submit: bool,
    resume: bool,
) -> Dict[str, Any]:
    config = load_yaml(config_path)
    collector_cfg = config.get("collector", {})
    sampling_cfg = config.get("sampling", {})
    schema = load_schema(collector_cfg.get("questionnaire_schema", "data/questionnaire/question_schema.json"))
    samples = generate_samples(
        schema,
        strategy=strategy,
        count=limit,
        seed=int(sampling_cfg.get("random_seed", 42)),
    )
    credentials = read_credentials()
    target_url = start_url or credentials.console_url or "https://play.google.com/console/developers"
    submit = bool(collector_cfg.get("default_submit", True)) and not dry_run and not no_submit
    samples_store = JsonlStore(collector_cfg.get("samples_path", "data/raw/samples.jsonl"))
    failed_store = JsonlStore(collector_cfg.get("failed_samples_path", "data/raw/failed_samples.jsonl"))
    existing_hashes = samples_store.answer_hashes() if resume else set()
    browser_config = BrowserConfig(
        profile_dir=collector_cfg.get("profile_dir", "browser_profile/play-console"),
        headless=bool(collector_cfg.get("headless", False)),
        navigation_timeout_ms=int(collector_cfg.get("navigation_timeout_ms", 60000)),
        selector_timeout_ms=int(collector_cfg.get("selector_timeout_ms", 15000)),
    )
    min_delay = int(collector_cfg.get("min_delay_seconds", 20))
    max_delay = int(collector_cfg.get("max_delay_seconds", 60))
    manual_every = int(collector_cfg.get("manual_review_every_n_samples", 50))
    manual_path = collector_cfg.get("manual_ops_path", "docs/人工操作清单.md")
    results = {"attempted": 0, "success": 0, "failed": 0, "dry_run": dry_run or no_submit}

    async with launch_persistent_context(browser_config) as context:
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(target_url, wait_until="domcontentloaded")
        for index, answers in enumerate(samples, start=1):
            from src.data.storage import answer_hash

            if resume and answer_hash(answers) in existing_hashes:
                continue
            record = await submit_sample(
                page=page,
                answers=answers,
                schema=schema,
                strategy=strategy,
                submit=submit,
                screenshots_dir=collector_cfg.get("screenshots_dir", "data/raw/screenshots"),
                html_dir=collector_cfg.get("html_dir", "data/raw/html"),
                manual_ops_path=manual_path,
            )
            results["attempted"] += 1
            if record.status == "success":
                results["success"] += 1
                samples_store.append(record.to_dict())
            else:
                results["failed"] += 1
                failed_store.append(record.to_dict())
                if record.status in {"blocked", "parse_error"}:
                    break
            if manual_every > 0 and index % manual_every == 0:
                append_manual_action(
                    "批量采集人工抽检",
                    f"已处理 {index} 条样本，请人工抽检 Play Console 状态和最近样本结果。",
                    path=manual_path,
                )
            if index < len(samples):
                delay = random.randint(min_delay, max_delay)
                await page.wait_for_timeout(delay * 1000)
    return results
