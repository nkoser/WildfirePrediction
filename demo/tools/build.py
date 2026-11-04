"""Assemble the demo from demo/web, demo/vendor and a data directory written by export_web.py:

    <out>/site/          the public site: index.html, data/ (manifest and layers), licenses/ (Leaflet, the
                         fonts), README.md, .nojekyll. Static files only, so any static host serves it as
                         is (GitHub Pages, a storage bucket).
    <out>/artifact/wildfire-survival-explorer.html
                         the same page as a fragment (title, style, body, scripts) for a preview host that
                         supplies its own <html>/<head>/<body> skeleton; publish it with <out>/site/data
                         next to it.

The page inlines Leaflet, the fonts and the basemap; it fetches only data/manifest.json and the layers the
manifest names, on demand. Before anything is written, the page and the manifest are checked for
  * any http(s) URL outside a short allow-list, so the page makes no third-party request;
  * local paths, and wording about the paper being under submission or review;
  * any pattern given with --forbid (names or wording that must stay off the public page).

    python -I demo/tools/build.py --data <export dir> [--out demo/dist] [--forbid PATTERN ...] [--noindex]
"""

from __future__ import annotations

import argparse
import base64
import math
import re
import shutil
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
TITLE = "Wildfire Survival Explorer"
DESCRIPTION = ("Interactive wildfire risk forecasts for Canada: for every 5 km pixel, the probability of a "
               "first fire within each lead time, with the fires observed afterwards.")
# browser-tab icon: a risk curve rising and leveling off, in the map's ramp colors
ICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" '
            'fill="#1a1c1e"/><path d="M6 25C12 25 13 8.5 26 7.5" fill="none" stroke="#fc8d59" stroke-width="3.2" '
            'stroke-linecap="round"/><circle cx="26" cy="7.5" r="2.6" fill="#fff7ec"/></svg>')
LICENSES = {"leaflet-1.9.4-LICENSE.txt": ("leaflet-1.9.4", "LICENSE"), "ibm-plex-OFL.txt": ("fonts", "OFL.txt")}
SITE_README = """# Wildfire Survival Explorer

Static files of an interactive map of wildfire risk forecasts. `index.html` is the page, with its scripts,
fonts and basemap inlined; `data/` holds the forecast manifest and one PNG per map layer. The files are
generated, so an edit made here is overwritten by the next build.

Third-party software and fonts ship under their own licenses, in `licenses/`: Leaflet (BSD-2-Clause) and
IBM Plex Sans (SIL Open Font License 1.1). The basemap is from Natural Earth (public domain).
"""
ALLOWED_URLS = {"https://leafletjs.com", "http://www.w3.org/2000/svg"}
FORBIDDEN = [r"under (double-blind )?(submission|review)", r"/mnt/", r"/home/", r"/raid/", r"/data/wildfire"]
# limits of the preview host, so the build says when the preview needs more than one publish
PREVIEW_FILE_LIMIT, PREVIEW_PUBLISH_BYTES, PREVIEW_PUBLISH_FILES = 15e6, 64e6, 255


def fonts_css() -> str:
    faces = [("IBM Plex Sans", "100 700", "ibm-plex-sans-latin.woff2"),
             ("IBM Plex Sans Condensed", "600", "ibm-plex-sans-condensed-600-latin.woff2")]
    out = []
    for family, weight, file in faces:
        b64 = base64.b64encode((DEMO / "vendor" / "fonts" / file).read_bytes()).decode("ascii")
        out.append(f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};font-display:swap;"
                   f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "\n".join(out)


def leaflet_css() -> str:
    css = (DEMO / "vendor" / "leaflet-1.9.4" / "leaflet.css").read_text()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)          # comments carry bug-tracker URLs
    return re.sub(r"url\(\s*['\"]?images/[^)]*\)", "none", css)   # marker and layer-switcher images, unused


def json_script(id_: str, text: str) -> str:
    if "</" in text:
        text = text.replace("</", "<\\/")
    return f'<script type="application/json" id="{id_}">{text}</script>'


def check(texts: dict[str, str], forbidden: list[str]) -> list[str]:
    problems = []
    for name, text in texts.items():
        visible = re.sub(r"data:[a-z0-9/+.-]+;base64,[A-Za-z0-9+/=]+", "data:", text)
        for url in sorted(set(re.findall(r"https?://[^\s\"'<>)\\]+", visible))):
            if url.rstrip("/.,;") not in ALLOWED_URLS:
                problems.append(f"{name}: external URL: {url}")
        for pat in forbidden:
            m = re.search(pat, visible, flags=re.IGNORECASE)
            if m:
                ctx = visible[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                problems.append(f"{name}: /{pat}/: ...{ctx}...")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="directory written by export_web.py")
    ap.add_argument("--basemap", type=Path, default=DEMO / "web" / "basemap.json")
    ap.add_argument("--out", type=Path, default=DEMO / "dist")
    ap.add_argument("--forbid", action="append", default=[], help="regular expression the page must not contain")
    ap.add_argument("--noindex", action="store_true", help="ask search engines not to index the page")
    args = ap.parse_args()
    manifest = args.data / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found; run export_web.py first")

    style = "\n".join([fonts_css(), leaflet_css(), (DEMO / "web" / "style.css").read_text()])
    body = (DEMO / "web" / "template.html").read_text()
    scripts = "\n".join([
        json_script("wf-basemap", args.basemap.read_text()),
        "<script>" + (DEMO / "vendor" / "leaflet-1.9.4" / "leaflet.js").read_text() + "</script>",
        "<script>" + (DEMO / "web" / "app.js").read_text() + "</script>",
    ])
    fragment = f"<title>{TITLE}</title>\n<style>\n{style}\n</style>\n{body}\n{scripts}\n"
    icon = "data:image/svg+xml;base64," + base64.b64encode(ICON_SVG.encode()).decode("ascii")
    head = "\n".join([
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{TITLE}</title>",
        f'<meta name="description" content="{DESCRIPTION}">',
        *(['<meta name="robots" content="noindex">'] if args.noindex else []),
        f'<link rel="icon" href="{icon}">',
    ])
    document = (f"<!doctype html>\n<html lang=\"en\">\n<head>\n{head}\n<style>\n{style}\n</style>\n</head>\n"
                f"<body>\n{body}\n{scripts}\n</body>\n</html>\n")

    problems = check({"page": document, "manifest": manifest.read_text()}, FORBIDDEN + args.forbid)
    if problems:
        raise SystemExit("build check failed:\n  " + "\n  ".join(problems))

    site = args.out / "site"
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(args.data, site / "data")
    (site / "index.html").write_text(document)
    (site / ".nojekyll").write_text("")
    (site / "README.md").write_text(SITE_README)
    for name, (folder, file) in LICENSES.items():
        (site / "licenses").mkdir(exist_ok=True)
        shutil.copyfile(DEMO / "vendor" / folder / file, site / "licenses" / name)
    art = args.out / "artifact" / "wildfire-survival-explorer.html"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(fragment)

    files = [p for p in (site / "data").rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    big = max(p.stat().st_size for p in files)
    print(f"[build] {site}: index.html {(site / 'index.html').stat().st_size / 1e6:.2f} MB, "
          f"data/ {len(files)} files, {total / 1e6:.1f} MB (largest {big / 1e6:.2f} MB)")
    print(f"[build] {art} {art.stat().st_size / 1e6:.2f} MB")
    publishes = max(math.ceil(total / PREVIEW_PUBLISH_BYTES), math.ceil(len(files) / PREVIEW_PUBLISH_FILES))
    if big > PREVIEW_FILE_LIMIT:
        print(f"[build] a data file is over the preview host's {PREVIEW_FILE_LIMIT / 1e6:.0f} MB per-file limit")
    elif publishes > 1:
        print(f"[build] the preview needs {publishes} publishes to hold data/ "
              f"(at most {PREVIEW_PUBLISH_FILES} files and {PREVIEW_PUBLISH_BYTES / 1e6:.0f} MB each)")
    print("[build] checks passed: no external URL outside the allow-list, no forbidden pattern"
          + ("; the page asks search engines not to index it" if args.noindex else ""))


if __name__ == "__main__":
    main()
