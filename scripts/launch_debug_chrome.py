#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir, project_path


DEFAULT_URL = "https://play.google.com/console/developers"


def candidate_chrome_paths() -> list[Path]:
    paths = []
    env_path = os.environ.get("PROGRAMFILES")
    if env_path:
        paths.append(Path(env_path) / "Google" / "Chrome" / "Application" / "chrome.exe")
    env_path_x86 = os.environ.get("PROGRAMFILES(X86)")
    if env_path_x86:
        paths.append(Path(env_path_x86) / "Google" / "Chrome" / "Application" / "chrome.exe")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        paths.append(Path(local_app_data) / "Google" / "Chrome" / "Application" / "chrome.exe")
    paths.append(Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    paths.append(Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"))
    return paths


def find_chrome_path(explicit: Optional[str]) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Chrome not found at: {candidate}")

    for candidate in candidate_chrome_paths():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Chrome automatically. Pass --chrome-path with the full chrome.exe path."
    )


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a dedicated debuggable Chrome for the questionnaire CDP probe."
    )
    parser.add_argument("--chrome-path", default=None)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--profile-dir", default="browser_profile/chrome-cdp")
    parser.add_argument("--fresh", action="store_true", help="Use a fresh temporary profile subdirectory.")
    return parser.parse_args()


def build_profile_dir(profile_dir: str, fresh: bool) -> Path:
    base = ensure_dir(project_path(profile_dir))
    if not fresh:
        return base

    index = 1
    while True:
        candidate = base / f"fresh_{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        index += 1


def main() -> None:
    args = parse_args()
    if port_is_open(args.port):
        print(f"Port {args.port} is already in use.")
        print(f"If that is your debuggable Chrome, you can directly run:")
        print(
            f".\\.venv\\Scripts\\python scripts\\probe_questionnaire_branches_cdp.py "
            f"--endpoint-url http://127.0.0.1:{args.port} --max-states 40"
        )
        return

    chrome_path = find_chrome_path(args.chrome_path)
    profile_dir = build_profile_dir(args.profile_dir, args.fresh)

    command = [
        str(chrome_path),
        f"--remote-debugging-port={args.port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
        args.url,
    ]

    subprocess.Popen(command)
    print(f"Launched Chrome: {chrome_path}")
    print(f"CDP endpoint should become available at: http://127.0.0.1:{args.port}")
    print(f"Profile directory: {profile_dir}")
    print("Log in inside that Chrome window first. Then run:")
    print(
        f".\\.venv\\Scripts\\python scripts\\probe_questionnaire_branches_cdp.py "
        f"--endpoint-url http://127.0.0.1:{args.port} --max-states 40"
    )


if __name__ == "__main__":
    main()
