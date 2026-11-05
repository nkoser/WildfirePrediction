# Interactive demo

The paper's public, interactive results page: a static site that goes online with the arXiv version, so
anyone can explore the forecasts, and later the front end of a hosted model.

**Status (2026-11-05):** the page shows real model output: `risk_map.py` maps from a Contagion-STS (three
seeds) for seven test-period forecast dates, every lead time from 1 to 30 days. Another model is a re-export
of its `risk_map.py` files with a new label; the page code does not change. `tools/make_placeholder.py`
makes stand-in data in the same format when no model output is at hand.

## What the page shows

- A map of F(τ), the probability that a pixel has its first MODIS fire detection within τ days after the
  forecast date, on the paper figure's log color scale (OrRd, 1e-4 up to the 99.9th percentile of F at
  the largest τ).
- A forecast-date picker and a lead-time slider through every exported τ (daily, 1 to 30). The slider reads
  the same forecast at every τ, which is the paper's point: one model, one curve, all lead times.
- Fires in the 30 input days (black) and fires after the forecast date within τ (blue), as in the paper's
  figure, drawn only inside the forecast area.
- Hover for the value under the cursor; click for the pixel's risk curve over all lead times, its values
  and the observed outcome. Hovering the curve reads a lead time; clicking it moves the map there.
- Following the paper's limitations, no predicted-arrival maps and no ignition/spread split.
- Everything fits a laptop window without scrolling (checked down to 660 px of window height at 1440 px
  wide, and at 1280 × 720): the panel holds the controls and the risk curve, which takes the height that is
  left; the explanations, the starting-location rule, the manifest's notes and the credits sit in the About
  dialog of the top bar.
- The top bar also holds the paper's arXiv link (a placeholder until `build.py --arxiv` names it) and a
  light/dark button. The reader's choice overrides the system theme and is kept in the browser's
  `localStorage`; choosing the system's own theme again drops the override.

## Layout

```
demo/
  grid.json      georeference of the raster (MPCFire grid, EPSG:4326, outer top-left corner)
  tools/         make_placeholder.py, make_basemap.py, export_web.py, build.py, check_page.py, common.py
  web/           template.html, style.css, app.js, basemap.json (Natural Earth 1:50m, generated)
  vendor/        Leaflet 1.9.4 (BSD-2-Clause), Montserrat and Roboto fonts (SIL OFL 1.1), with their licenses
  dist/          build output, not in git: site/ (the deployable site)
```

## Build

Python 3.12 with numpy and pillow; the check also needs playwright with Chromium.

```bash
# 1. data directory from risk_map.py outputs (or from make_placeholder.py, see below)
python -I demo/tools/export_web.py /data/Wildfire/figures/risk_maps/risk_cg2w_ema_*.npz \
    --model-id contagion-sts --model-label Contagion-STS \
    --model-detail "ensemble of 3 seeds, raw probabilities" --out work/data
# 2. site: dist/site/ (index.html, data/ copied from step 1, .nojekyll, licenses/, README.md);
#    --arxiv https://arxiv.org/abs/<id> turns the top bar's placeholder into the paper's link
python -I demo/tools/build.py --data work/data
# 3. serve dist/site on 127.0.0.1 and drive it in headless Chromium; fails on any script error,
#    failed request or request to another origin, or when the page does not respond as expected
python -I demo/tools/check_page.py demo/dist/site --shots work/shots
```

To look at it locally: `python3 -m http.server -d demo/dist/site` and open the printed address. The page
fetches its data, so opening `index.html` from disk does not work.

Placeholder inputs: `make_placeholder.py --mcd14a1 <dir with MCD14A1__<year>.tar> --naturalearth <dir> --out work/placeholder`.
Basemap: `make_basemap.py --naturalearth <dir with ne_50m_*.geojson> --out demo/web/basemap.json`.

## Hosting

`dist/site` is static files only, so GitHub Pages serves it as is (`.nojekyll` is included), as does any
static host or storage bucket.

- **Page:** about 1 MB, with Leaflet, the fonts and the basemap inlined.
- **Data:** a risk layer is about 0.3 MB (0.17 to 0.38 MB for the seven dates), so a forecast date with 30
  lead times is about 9 MB and the seven dates 63 MB.
- **Loading:** the first view needs the page, the manifest, the observed fires and three layers (the chosen
  lead time, plus the first and last to pick the starting location), about 1.4 MB; the date's other lead
  times load right after, four at a time, for the risk curve, about 9.3 MB for the whole date.
- **GitHub Pages limits** (GitHub docs, read 2026-11-04): 1 GB per published site, 1 GB recommended per
  repository, a soft 100 GB per month of bandwidth. Daily live forecasts at about 9 MB a day should not
  pile up in a git repository; they belong with the service that makes them (see below).

## Data in: risk-map files

One `.npz` per model and forecast date, as written by `wildfire/paper/risk_map.py`:

| key | type | meaning |
|---|---|---|
| `F_<tau>` | float32 (H, W) | pixel probability of a first fire within τ days after the date; NaN outside the forecast area |
| `fire_window` | bool (H, W) | MODIS fire in the input days L-29 .. L |
| `first_fire` | int (H, W) | first MODIS fire day after L (1 .. max τ), 0 = none |
| `taus`, `date` | | lead times and forecast date L |

`ign_<tau>`, `fire_today` and the rest are ignored. H × W and the georeference must match `grid.json`.

## Data out: the data directory (schema `wildfire-demo/2`)

`export_web.py` writes `manifest.json` and, per model and date, `risk-<tau>.png` and `observed.png` under
`<model>/<date>/`. Every path in the manifest is relative to the manifest's URL, and the page reads nothing
else, so a hosted model that publishes the same directory, or serves the same files with CORS, drives the
page without code changes. This is the draft contract for that service.

| field | meaning |
|---|---|
| `grid` | CRS, width, height, outer top-left corner (`lon0`, `lat0`), pixel size `res` in degrees |
| `quant` | risk layers are 8-bit grayscale PNG: 1..255 on a log10 scale from `vmin` to `vmax`, 0 = no data |
| `display` | color-scale range |
| `models` | `id`, `label`, `detail` |
| `forecasts[]` | `model`, `date`, `taus`, `risk[tau]` (path of a PNG), `observed` (path of a PNG: bit 7 = fire in the input window, bits 0-6 = first fire day after the date), `observed_days` |
| `placeholder` | true when any input was a placeholder; the page then shows a banner |
| `notes` | notes on the model and data, shown in the About dialog |
| `default` | date and lead time the page opens on |

## Checks

- The page is self-contained: every script, font and the basemap ship with it, and it requests nothing
  but its own data. `build.py` fails on an http(s) URL outside a short allow-list (Leaflet's license
  header, the SVG namespace); `check_page.py` fails on any request to another origin.
- `build.py` also fails on local paths and on wording about the paper's review status in the page or the
  manifest; `--forbid PATTERN` adds patterns of its own.
- `check_page.py` drives the built site in headless Chromium: opening view, every lead time loaded, date
  switch, slider, map click, curve hover, theme button (including a reload), About dialog, and a panel that
  fits without scrolling at 1440 × 900 and at `--small` (default 1280 × 720). It prints the lowest window
  height at which the panel still fits.
