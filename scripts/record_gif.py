#!/usr/bin/env python
"""Record the live Streamlit dashboard to a GIF (headless Chromium + Pillow).

    python scripts/record_gif.py [--out results/dashboard_demo.gif] [--seconds 24] [--fps 2]

The GIF shows whatever device the dashboard runs on (it is printed on the page). Re-record it
on the RTX 4060 for the README; a GIF recorded on a CPU host says "Device: CPU".
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _launch(p):
    try:
        return p.chromium.launch()
    except Exception as first:  # noqa: BLE001
        root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))
        for exe in sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")), reverse=True):
            try:
                return p.chromium.launch(executable_path=exe)
            except Exception:  # noqa: BLE001
                continue
        raise SystemExit(f"cannot launch Chromium ({str(first).splitlines()[0]}); run: playwright install chromium")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dashboard_demo.gif")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=900)
    args = ap.parse_args()

    from PIL import Image
    from playwright.sync_api import sync_playwright

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"), "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = []
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                if urllib.request.urlopen(url + "/_stcore/health", timeout=2).status == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        with sync_playwright() as p:
            browser = _launch(p)
            page = browser.new_page(viewport={"width": 1400, "height": 1180})
            page.goto(url, timeout=60_000)
            page.get_by_role("heading", name=re.compile(r"Left Hand|Right Hand")).first.wait_for(timeout=180_000)
            page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length >= 2", timeout=30_000)
            time.sleep(8)                                    # let the latency window and the plot fill
            t_end = time.time() + args.seconds
            while time.time() < t_end:
                t0 = time.time()
                img = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
                h = round(img.height * args.width / img.width)
                frames.append(img.resize((args.width, h), Image.LANCZOS).quantize(colors=64, dither=Image.Dither.NONE))
                time.sleep(max(0.0, 1 / args.fps - (time.time() - t0)))
            browser.close()
    finally:
        proc.terminate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
