"""Serve the built site over HTTP on 127.0.0.1 and drive it in headless Chromium at a laptop size.

Fails on any console error or warning, page error, failed request, or request to another origin, and when
the page does not behave: it must open on the manifest's default view, load every lead time of the shown
date (the risk curve then has one point per lead time), switch date, move to the longest lead time, and
select a location on a click. Prints how much the first view and the full date download, and writes
screenshots of both themes and of the changed state.

    PLAYWRIGHT_BROWSERS_PATH=<browsers> python -I demo/tools/check_page.py demo/dist/site --shots <dir>
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def serve(root: Path) -> http.server.ThreadingHTTPServer:
    """Threaded, so the page's parallel layer requests do not queue behind each other."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def curve_points(page) -> int:
    pts = page.locator("#curve polyline.line").get_attribute("points") or ""
    return len(pts.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("site", type=Path, help="the built site directory (index.html, data/)")
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()
    args.shots.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.site / "data" / "manifest.json").read_text())
    dates = sorted({f["date"] for f in manifest["forecasts"]})
    default = manifest["default"]["date"] if manifest["default"]["date"] in dates else dates[0]
    n_taus = {f["date"]: len(f["taus"]) for f in manifest["forecasts"]}
    srv = serve(args.site)
    origin = f"http://127.0.0.1:{srv.server_address[1]}"
    errors, foreign, failed, problems = [], [], [], []
    loads: list[tuple[str, int]] = []

    def expect(ok: bool, what: str) -> None:
        print(f"[check] {'ok  ' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            problems.append(what)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for scheme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": args.width, "height": args.height}, color_scheme=scheme)
            page = ctx.new_page()
            page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
            page.on("request", lambda r: foreign.append(r.url)
                    if not r.url.startswith((origin + "/", "data:", "blob:")) else None)
            page.on("requestfailed", lambda r: failed.append(f"{r.url}: {r.failure}"))
            page.on("response", lambda r: loads.append((r.url, int(r.headers.get("content-length", 0)))))
            loads.clear()
            page.goto(origin + "/")
            page.wait_for_function("window.__wfReady === true", timeout=60000)
            first_view = (len(loads), sum(b for _, b in loads))
            page.wait_for_function(f"window.__wfComplete === '{default}'", timeout=120000)
            full = (len(loads), sum(b for _, b in loads))
            page.wait_for_timeout(400)
            page.screenshot(path=str(args.shots / f"{scheme}.png"))
            if scheme == "dark":
                ctx.close()
                continue
            print(f"[check] first view: {first_view[0]} requests, {first_view[1] / 2**20:.1f} MB; "
                  f"all lead times of {default}: {full[0]} requests, {full[1] / 2**20:.1f} MB", flush=True)
            expect(not page.locator("#error-note").is_visible(), "no error note")
            expect(curve_points(page) == n_taus[default],
                   f"curve has {n_taus[default]} points after loading ({curve_points(page)})")
            print("[check] opening view:", page.locator("#map-title").inner_text(), "|",
                  page.locator("#loc-name").inner_text(), flush=True)
            other = dates[2] if len(dates) > 2 else dates[-1]
            page.locator(".date-btn").nth(dates.index(other)).click()   # buttons are in date order
            page.wait_for_function(f"window.__wfComplete === '{other}'", timeout=120000)
            expect(curve_points(page) == n_taus[other], f"curve has {n_taus[other]} points after switching to {other}")
            page.locator("#lead").fill(page.locator("#lead").get_attribute("max"))
            page.wait_for_function("document.getElementById('map-title').getAttribute('aria-busy') === 'false'")
            title = page.locator("#map-title").inner_text()
            expect(f"{max(manifest['forecasts'][0]['taus'])} days" in title, f"longest lead time shown ({title})")
            before = after = page.locator("#loc-name").inner_text()
            for fx, fy in ((0.25, 0.40), (0.33, 0.45), (0.50, 0.40), (0.62, 0.45)):   # until one lands on forecast land
                page.mouse.click(args.width * fx, args.height * fy)
                page.wait_for_timeout(300)
                after = page.locator("#loc-name").inner_text()
                if after != before:
                    break
            expect(after != before, f"a click on the map selects that location ({before} -> {after})")
            box = page.locator("#curve svg").bounding_box()
            page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
            page.wait_for_timeout(200)
            expect(page.locator("#curve .hover").get_attribute("visibility") == "visible", "curve hover shows a value")
            page.locator("#curve").screenshot(path=str(args.shots / "curve-hover.png"))
            page.mouse.move(args.width * 0.35, args.height * 0.45)
            page.wait_for_timeout(600)
            page.screenshot(path=str(args.shots / "changed.png"))
            ctx.close()
        browser.close()
    srv.shutdown()
    for e in errors:
        print("[check]", e)
    for u in foreign:
        print("[check] request to another origin:", u)
    for f in failed:
        print("[check] failed request:", f)
    if errors or foreign or failed or problems:
        sys.exit(1)
    print(f"[check] passed: no errors, no failed or foreign requests; screenshots in {args.shots}")


if __name__ == "__main__":
    main()
