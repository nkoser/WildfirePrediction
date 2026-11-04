"""Assemble the demo page from demo/web, demo/vendor and a data manifest, in two variants:

    <out>/supplement/index.html   a complete, self-contained document for the supplementary zip;
                                  opens from file:// and makes no request to any server
    <out>/artifact/wildfire-survival-explorer.html
                                  the same page as a fragment (title, style, body, scripts) for a host that
                                  supplies its own <html>/<head>/<body> skeleton

Both inline everything: Leaflet, fonts, basemap and data. Then two checks run on the supplement file and
the build fails if either finds something:
  * external references: any http(s) URL outside a short allow-list (Leaflet's license header, the SVG
    namespace), since the venue forbids links that expand the submission and a request would reveal reviewers;
  * identifying strings: names, repository, institution and cluster paths, for double-blind review.

    python -I demo/tools/build.py --data <data.json> [--basemap demo/web/basemap.json] [--out demo/dist]
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
TITLE = "Wildfire Survival Explorer"
ALLOWED_URLS = {"https://leafletjs.com", "http://www.w3.org/2000/svg"}
IDENTIFYING = [r"koser", r"niklas", r"bartkowiak", r"patryk", r"wildfireprediction", r"github\.com",
               r"/mnt/", r"/home/", r"/data/wildfire", r"/raid/", r"vcai", r"pcss", r"pozna", r"\bamu\b",
               r"\beagle\b", r"lübeck|luebeck", r"overleaf", r"anonymousdata4neurips"]
ARTIFACT_LIMIT = 16 * 2**20


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


def check(html: str) -> list[str]:
    problems = []
    visible = re.sub(r"data:[a-z0-9/+.-]+;base64,[A-Za-z0-9+/=]+", "data:", html)
    for url in sorted(set(re.findall(r"https?://[^\s\"'<>)\\]+", visible))):
        if url.rstrip("/.,;") not in ALLOWED_URLS:
            problems.append(f"external URL: {url}")
    for pat in IDENTIFYING:
        for m in re.finditer(pat, visible, flags=re.IGNORECASE):
            ctx = visible[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
            problems.append(f"identifying string /{pat}/: ...{ctx}...")
            break
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--basemap", type=Path, default=DEMO / "web" / "basemap.json")
    ap.add_argument("--out", type=Path, default=DEMO / "dist")
    args = ap.parse_args()

    style = "\n".join([fonts_css(), leaflet_css(), (DEMO / "web" / "style.css").read_text()])
    body = (DEMO / "web" / "template.html").read_text()
    scripts = "\n".join([
        json_script("wf-data", args.data.read_text()),
        json_script("wf-basemap", args.basemap.read_text()),
        "<script>" + (DEMO / "vendor" / "leaflet-1.9.4" / "leaflet.js").read_text() + "</script>",
        "<script>" + (DEMO / "web" / "app.js").read_text() + "</script>",
    ])
    fragment = f"<title>{TITLE}</title>\n<style>\n{style}\n</style>\n{body}\n{scripts}\n"
    document = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                f"<title>{TITLE}</title>\n<style>\n{style}\n</style>\n</head>\n<body>\n{body}\n{scripts}\n</body>\n</html>\n")

    problems = check(document)
    if problems:
        raise SystemExit("build check failed:\n  " + "\n  ".join(problems))
    sup = args.out / "supplement" / "index.html"
    art = args.out / "artifact" / "wildfire-survival-explorer.html"
    for path, text in ((sup, document), (art, fragment)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    size = art.stat().st_size
    print(f"[build] {sup} {sup.stat().st_size / 2**20:.2f} MB")
    print(f"[build] {art} {size / 2**20:.2f} MB" + (" - OVER the 16 MB artifact limit" if size > ARTIFACT_LIMIT else ""))
    print("[build] checks passed: no external URL outside the allow-list, no identifying string")


if __name__ == "__main__":
    main()
