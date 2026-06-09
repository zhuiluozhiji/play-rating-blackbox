from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from src.common import ensure_dir, project_path


@dataclass
class BrowserConfig:
    profile_dir: str = "browser_profile/play-console"
    headless: bool = False
    navigation_timeout_ms: int = 60000
    selector_timeout_ms: int = 15000


@asynccontextmanager
async def launch_persistent_context(config: BrowserConfig) -> AsyncIterator[object]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements.txt` and "
            "`python -m playwright install chromium`."
        ) from exc

    profile_dir = ensure_dir(config.profile_dir)
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=config.headless,
            viewport={"width": 1440, "height": 1000},
            accept_downloads=False,
        )
        context.set_default_timeout(config.selector_timeout_ms)
        context.set_default_navigation_timeout(config.navigation_timeout_ms)
        try:
            yield context
        finally:
            await context.close()
