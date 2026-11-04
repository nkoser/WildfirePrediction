# Interactive demo

An anonymous, self-contained results page for the supplementary material, and later the public demo
backed by a hosted model.

**Status (2026-11-04):** the page runs on placeholder risk values (`tools/make_placeholder.py`) with the real
MODIS fires of MPCFire. It switches to model output by re-running the export on the `risk_map.py` files;
no code changes.

## What the page shows

- A map of F(τ), the probability that a pixel has its first MODIS fire detection within τ days after the
  forecast date, on the paper figure's log color scale (OrRd, 1e-4 up to the 99.9th percentile of F at
  the largest τ).
- A forecast-date picker and a lead-time slider. The slider reads the same forecast at every τ in the
  export, which is the paper's point: one model, one curve, all lead times.
- Fires in the 30 input days (black) and fires after the forecast date within τ (blue), as in the paper's
  figure, drawn only inside the forecast area.
- Hover for the value under the cursor; click for the pixel's risk curve, its values and the observed
  outcome.
- Following the paper's limitations, no predicted-arrival maps and no ignition/spread split.

## Layout

```
demo/
  grid.json      georeference of the raster (MPCFire grid, EPSG:4326, outer top-left corner)
  tools/         make_placeholder.py, make_basemap.py, export_web.py, build.py, check_page.py, common.py
  web/           template.html, style.css, app.js, basemap.json (Natural Earth 1:50m, generated)
  vendor/        Leaflet 1.9.4 (BSD-2-Clause), IBM Plex Sans fonts (SIL OFL 1.1), with their licenses
  dist/          build output, not in git
```

## Build

Python 3.12 with numpy and pillow; the check also needs playwright with Chromium.

```bash
# 1. data manifest from risk_map.py outputs (or from make_placeholder.py, see below)
python -I demo/tools/export_web.py /data/Wildfire/figures/risk_maps/risk_cg2w_ema_*.npz \
    --model-id contagion-sts --model-label Contagion-STS \
    --model-detail "ensemble of 3 seeds, raw probabilities" --out work/data.json
# 2. page: dist/supplement/index.html (complete document) and dist/artifact/*.html (fragment)
python -I demo/tools/build.py --data work/data.json
# 3. open it from file:// in headless Chromium; fails on any script error or network request
python -I demo/tools/check_page.py demo/dist/supplement/index.html --shots work/shots
```

Placeholder inputs: `make_placeholder.py --mcd14a1 <dir with MCD14A1__<year>.tar> --naturalearth <dir> --out work/placeholder`.
Basemap: `make_basemap.py --naturalearth <dir with ne_50m_*.geojson> --out demo/web/basemap.json`.

## Data in: risk-map files

One `.npz` per model and forecast date, as written by `wildfire/paper/risk_map.py`:

| key | type | meaning |
|---|---|---|
| `F_<tau>` | float32 (H, W) | pixel probability of a first fire within τ days after the date; NaN outside the forecast area |
| `fire_window` | bool (H, W) | MODIS fire in the input days L-29 .. L |
| `first_fire` | int (H, W) | first MODIS fire day after L (1 .. max τ), 0 = none |
| `taus`, `date` | | lead times and forecast date L |

`ign_<tau>`, `fire_today` and the rest are ignored. H × W and the georeference must match `grid.json`.

## Data out: the manifest (schema `wildfire-demo/1`)

`export_web.py` writes one JSON document. The page reads nothing else, so the same document, with layer
URLs in place of embedded images, is the draft contract for a hosted model.

| field | meaning |
|---|---|
| `grid` | CRS, width, height, outer top-left corner (`lon0`, `lat0`), pixel size `res` in degrees |
| `quant` | risk layers are 8-bit grayscale PNG: 1..255 on a log10 scale from `vmin` to `vmax`, 0 = no data |
| `display` | color-scale range |
| `models` | `id`, `label`, `detail` |
| `forecasts[]` | `model`, `date`, `taus`, `risk[tau]` (PNG), `observed` (PNG: bit 7 = fire in the input window, bits 0-6 = first fire day after the date), `observed_days` |
| `placeholder` | true when any input was a placeholder; the page then shows a banner |
| `notes` | footnotes shown under the panel |

Size: about 150 KB per risk layer, so 4 dates × 5 lead times build to 4.9 MB.

## Rules for the supplement copy

- No external link or request: the venue's author guidelines forbid links that expand the submission, and a
  request to a server would reveal when reviewers open the page. Everything is inlined; the page opens from
  disk with no server.
- No identifying names, institutions, repositories or paths.
- `build.py` fails if the page contains an http(s) URL outside a short allow-list (Leaflet's license
  header, the SVG namespace) or an identifying string; `check_page.py` fails on any network request.
