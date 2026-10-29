"""Headless-browser test: the Streamlit dashboard renders, shows a class, a latency gauge and
live-updating EEG without raising an exception (test-matrix row 4)."""

from __future__ import annotations

import glob
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api", reason="pip install playwright (dev extra)")

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch_chromium(p):
    try:
        return p.chromium.launch()
    except Exception as first:  # noqa: BLE001 - e.g. browser build differs from the pinned playwright revision
        root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))
        for exe in sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")), reverse=True):
            try:
                return p.chromium.launch(executable_path=exe)
            except Exception:  # noqa: BLE001
                continue
        pytest.fail(f"cannot launch Chromium ({str(first).splitlines()[0]}). Run: playwright install chromium",
                    pytrace=False)


@pytest.fixture(scope="module")
def dashboard_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"), "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if urllib.request.urlopen(url + "/_stcore/health", timeout=2).status == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            pytest.fail("streamlit did not start within 60 s")
        yield url
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=15)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0]
        assert "Traceback" not in out, f"streamlit logged a traceback:\n{out[-3000:]}"


def _frames(page) -> int:
    text = page.get_by_text(re.compile(r"Chunks processed: \d+")).first.inner_text()
    return int(re.search(r"Chunks processed: (\d+)", text).group(1))


def test_dashboard_renders_and_updates(dashboard_url):
    with playwright_sync.sync_playwright() as p:
        browser = _launch_chromium(p)
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(dashboard_url, timeout=60_000)

        # engine process trains + warms up first (tens of seconds on CPU), then the live view appears
        page.get_by_text("Predicted class").first.wait_for(timeout=120_000)
        page.get_by_role("heading", name=re.compile(r"Left Hand|Right Hand")).first.wait_for(timeout=120_000)
        page.get_by_text(re.compile(r"p99 latency")).first.wait_for(timeout=30_000)
        page.get_by_text("Raw EEG, 64 channels").first.wait_for(timeout=30_000)
        page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length >= 2", timeout=30_000)

        first = _frames(page)
        time.sleep(3)
        second = _frames(page)
        assert second > first, f"dashboard is not live: chunks processed {first} -> {second}"
        assert page.locator('[data-testid="stException"]').count() == 0
        assert not errors, f"browser page errors: {errors}"
        assert page.get_by_text(re.compile(r"Device: ")).count() >= 1
        browser.close()
