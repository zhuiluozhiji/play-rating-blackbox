#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir, write_json

from probe_questionnaire_branches import (
    ProbeConfig,
    build_question_catalog,
    first_unanswered_question,
    path_id,
    replay_path,
    save_state_artifacts,
    summarize_path,
    wait_after_action,
)


def fetch_json(url: str) -> Any:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "play-rating-blackbox-cdp-probe",
        },
    )
    # Local CDP endpoints should bypass system/user HTTP proxies.
    if hostname in {"127.0.0.1", "localhost"}:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        response = opener.open(request, timeout=3)
    else:
        response = urllib.request.urlopen(request, timeout=3)
    with response:
        return json.loads(response.read().decode("utf-8"))


def endpoint_available(endpoint_url: str) -> bool:
    try:
        if endpoint_url.startswith("ws://") or endpoint_url.startswith("wss://"):
            return True
        fetch_json(endpoint_url.rstrip("/") + "/json/version")
        return True
    except Exception:
        return False


def format_attach_help(endpoint_url: str) -> str:
    return (
        f"Could not connect to Chrome DevTools at {endpoint_url}.\n"
        "This usually means your currently opened Chrome was not started with remote debugging enabled.\n"
        "\n"
        "Important limitation:\n"
        "- A normal already-open Chrome window usually cannot be attached unless it already exposes a CDP endpoint.\n"
        "- On newer Chrome versions, remote debugging switches are restricted for the default Chrome data directory.\n"
        "\n"
        "What this script needs:\n"
        "- A Chrome/Chromium instance that already exposes a DevTools endpoint, usually via --remote-debugging-port.\n"
        "\n"
        "If you want, I can next give you a one-command launcher that starts a debuggable Chrome window for this lab."
    )


async def prompt(message: str) -> str:
    return await asyncio.to_thread(input, message)


def collect_pages(browser: Any) -> List[Any]:
    pages: List[Any] = []
    for context in browser.contexts:
        pages.extend(context.pages)
    return pages


async def choose_page(
    browser: Any,
    target_substring: str,
    page_index: Optional[int] = None,
    prompt_user: bool = True,
) -> Any:
    pages = collect_pages(browser)
    if not pages:
        raise RuntimeError("Connected to Chrome, but no attachable pages were found.")

    descriptions = []
    default_index = None
    lowered_target = target_substring.lower()
    for index, page in enumerate(pages):
        url = page.url or ""
        title = ""
        try:
            title = await page.title()
        except Exception:
            title = ""
        descriptions.append({"index": index, "url": url, "title": title})
        if default_index is None and lowered_target and lowered_target in url.lower():
            default_index = index

    if default_index is None:
        for item in reversed(descriptions):
            if item["url"] and item["url"] != "about:blank":
                default_index = item["index"]
                break
    if default_index is None:
        default_index = 0

    print("Detected Chrome pages:")
    for item in descriptions:
        marker = "*" if item["index"] == default_index else " "
        print(f"  {marker} [{item['index']}] {item['title'][:80]}  {item['url']}")

    if page_index is not None:
        chosen_index = page_index
    elif prompt_user:
        answer = (await prompt(f"Choose page index to probe [default {default_index}]: ")).strip()
        if answer:
            try:
                chosen_index = int(answer)
            except ValueError as exc:
                raise RuntimeError(f"Invalid page index: {answer}") from exc
        else:
            chosen_index = default_index
    else:
        chosen_index = default_index

    if chosen_index < 0 or chosen_index >= len(pages):
        raise RuntimeError(f"Page index out of range: {chosen_index}")
    return pages[chosen_index]


async def run_probe_via_cdp(
    endpoint_url: str,
    target_substring: str,
    config: ProbeConfig,
    page_index: Optional[int] = None,
    assume_ready: bool = False,
) -> Dict[str, Any]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed in the current environment.") from exc

    ensure_dir(config.output_dir)
    states: List[Dict[str, Any]] = []
    queue: deque[List[Dict[str, Any]]] = deque([[]])
    queued = {path_id([])}
    processed = set()

    async with async_playwright() as playwright:
        connect_target = endpoint_url
        if endpoint_url.startswith("http://") or endpoint_url.startswith("https://"):
            version_info = fetch_json(endpoint_url.rstrip("/") + "/json/version")
            connect_target = version_info.get("webSocketDebuggerUrl") or endpoint_url
        browser = await playwright.chromium.connect_over_cdp(connect_target, no_defaults=True)
        page = await choose_page(
            browser,
            target_substring,
            page_index=page_index,
            prompt_user=not assume_ready and page_index is None,
        )
        print(f"Using page: {page.url}")
        if not assume_ready:
            print("Bring that tab to the questionnaire page in Chrome, then come back here.")
            await prompt("When the questionnaire page is ready, press Enter to start probing...")
        await wait_after_action(page, config.settle_ms)
        start_url = page.url
        if not start_url or start_url == "about:blank":
            raise RuntimeError("The selected page is still about:blank. Open the questionnaire page first.")

        while queue and len(states) < config.max_states:
            path = queue.popleft()
            current_path_id = path_id(path)
            queued.discard(current_path_id)
            if current_path_id in processed:
                continue
            processed.add(current_path_id)
            parent_state_id = path_id(path[:-1]) if path else None

            try:
                snapshot = await replay_path(
                    page,
                    start_url,
                    path,
                    config.settle_ms,
                    fallback_email=config.fallback_email,
                )
                artifacts = await save_state_artifacts(
                    page=page,
                    output_dir=config.output_dir,
                    state_id=current_path_id,
                    snapshot=snapshot,
                    screenshot=config.screenshot,
                )
                error = None
            except Exception as exc:
                snapshot = {
                    "url": page.url,
                    "title": "",
                    "headings": [],
                    "questions": [],
                    "auxiliary_controls": [],
                    "buttons": [],
                    "errors": [],
                    "body_fingerprint": "",
                    "state_signature": "",
                    "can_continue": False,
                    "can_finalize": False,
                }
                artifacts = {}
                error = str(exc)

            state = {
                "state_id": current_path_id,
                "parent_state_id": parent_state_id,
                "path": path,
                "path_summary": summarize_path(path),
                "path_depth": len(path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "snapshot": snapshot,
                "artifacts": artifacts,
                "error": error,
            }
            states.append(state)
            print(f"[{len(states):03d}] {current_path_id} questions={len(snapshot['questions'])} error={bool(error)}")

            if error:
                continue

            next_question = first_unanswered_question(snapshot, path)
            if next_question:
                for option in next_question["options"][: config.max_options_per_question]:
                    child_path = path + [
                        {
                            "kind": "answer",
                            "question_key": next_question["question_key"],
                            "question_text": next_question["text"],
                            "option_key": option["option_key"],
                            "option_label": option["label"],
                        }
                    ]
                    child_id = path_id(child_path)
                    if child_id not in queued and child_id not in processed:
                        queue.append(child_path)
                        queued.add(child_id)
                continue

            if snapshot["can_continue"] and not snapshot["can_finalize"]:
                child_path = path + [{"kind": "continue"}]
                child_id = path_id(child_path)
                if child_id not in queued and child_id not in processed:
                    queue.append(child_path)
                    queued.add(child_id)

        catalog = build_question_catalog(states)
        edges = []
        by_id = {state["state_id"]: state for state in states}
        for state in states:
            if not state.get("parent_state_id"):
                continue
            parent = by_id.get(state["parent_state_id"])
            current_keys = {item["question_key"] for item in state["snapshot"].get("questions", [])}
            parent_keys = {item["question_key"] for item in parent["snapshot"].get("questions", [])} if parent else set()
            edges.append(
                {
                    "from_state_id": state["parent_state_id"],
                    "to_state_id": state["state_id"],
                    "action": state["path"][-1] if state["path"] else None,
                    "new_question_keys": sorted(current_keys - parent_keys),
                    "removed_question_keys": sorted(parent_keys - current_keys),
                }
            )

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "endpoint_url": endpoint_url,
            "start_url": start_url,
            "state_count": len(states),
            "question_count": catalog["question_count"],
            "states_path": str(config.output_dir / "states.json"),
            "edges_path": str(config.output_dir / "edges.json"),
            "question_catalog_path": str(config.output_dir / "question_catalog.json"),
        }
        write_json(config.output_dir / "states.json", states)
        write_json(config.output_dir / "edges.json", edges)
        write_json(config.output_dir / "question_catalog.json", catalog)
        write_json(config.output_dir / "summary.json", summary)
        await browser.close()
        return summary


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "probes" / "cdp" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach to an existing Chrome DevTools endpoint and probe questionnaire branches."
    )
    parser.add_argument("--endpoint-url", default="http://127.0.0.1:9222")
    parser.add_argument("--target-substring", default="play.google.com/console")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--page-index", type=int, default=None)
    parser.add_argument("--max-states", type=int, default=300)
    parser.add_argument("--max-options-per-question", type=int, default=8)
    parser.add_argument("--max-multi-combinations", type=int, default=32)
    parser.add_argument("--settle-ms", type=int, default=1200)
    parser.add_argument("--no-screenshot", action="store_true")
    parser.add_argument("--fallback-email", default="")
    parser.add_argument("--assume-ready", action="store_true")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if not endpoint_available(args.endpoint_url):
        raise SystemExit(format_attach_help(args.endpoint_url))

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    config = ProbeConfig(
        start_url="",
        output_dir=output_dir,
        max_states=args.max_states,
        max_options_per_question=args.max_options_per_question,
        max_multi_combinations=args.max_multi_combinations,
        settle_ms=args.settle_ms,
        screenshot=not args.no_screenshot,
        fallback_email=args.fallback_email,
    )
    summary = await run_probe_via_cdp(
        endpoint_url=args.endpoint_url,
        target_substring=args.target_substring,
        config=config,
        page_index=args.page_index,
        assume_ready=args.assume_ready,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
