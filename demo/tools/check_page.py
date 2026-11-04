"""Open the built supplement page the way a reviewer would (file://, no server) in headless Chromium.

Fails on any JavaScript error or any request that is not file:, data: or blob:. Writes screenshots of
the light and dark themes and of a changed state (later date, longest lead time) at a laptop size.

    PLAYWRIGHT_BROWSERS_PATH=<browsers> python -I demo/tools/check_page.py demo/dist/supplement/index.html --shots <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page", type=Path)
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()
    args.shots.mkdir(parents=True, exist_ok=True)
    url = args.page.resolve().as_uri()
    errors, external = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for scheme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": args.width, "height": args.height}, color_scheme=scheme)
            page = ctx.new_page()
            page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
            page.on("request", lambda r: external.append(r.url) if not r.url.startswith(("file:", "data:", "blob:")) else None)
            page.goto(url)
            page.wait_for_function("window.__wfReady === true", timeout=30000)
            page.wait_for_timeout(600)
            page.screenshot(path=str(args.shots / f"{scheme}.png"))
            if scheme == "light":
                print("[check] error note:", page.locator("#error-note").is_visible(),
                      "| title:", page.locator("#map-title").inner_text(),
                      "| location:", page.locator("#loc-name").inner_text(), flush=True)
                page.locator(".date-btn").nth(2).click()
                page.locator("#lead").fill(str(int(page.locator("#lead").get_attribute("max"))))
                page.mouse.move(args.width * 0.35, args.height * 0.45)
                page.wait_for_timeout(800)
                page.screenshot(path=str(args.shots / "changed.png"))
                print("[check] after change:", page.locator("#map-title").inner_text(), flush=True)
            ctx.close()
        browser.close()
    for e in errors:
        print("[check]", e)
    for u in external:
        print("[check] external request:", u)
    if errors or external:
        sys.exit(1)
    print(f"[check] ok: no errors, no external requests; screenshots in {args.shots}")


if __name__ == "__main__":
    main()
