# Live inputs: MPCFire's channels rebuilt from Earth Engine

The model reads MPCFire's daily rasters (55 channels on a 1968 x 928 grid of 0.0449 degrees, uint8). MPCFire
publishes them up to 2025 only. This package rebuilds them for any date from Google Earth Engine with the recipe
of the FireMPC code (github.com/SynUW/FireMPC), as it was run, and checks every channel against MPCFire's released
2025 rasters. The choices the code leaves open (arguments that were not recorded, a missing shapefile, Earth
Engine behavior) were measured against those rasters.

Status as of 2026-11-05: the fire channel, ERA5-Land (18 bands) and VPD are rebuilt exactly; FWI is close; the MODIS
surface channels are approximate (simple rules, not calibrated to the code). The model does not notice the
difference beyond the acceptance thresholds (below), and the demo shows seven forecasts of 2026-10-24 .. 10-30.

## Modules

| module | does |
|---|---|
| `grid.py` | the release grid, Earth Engine's 1 km grid it was cut from, the 0.1-degree ERA5-Land grid |
| `ee_io.py` | Earth Engine sign-in (`EE_PROJECT`, workload tag `wildfire-demo-live`) and pixel downloads on a fixed grid |
| `encode.py` | MPCFire's float-to-uint8 encoding and its per-band ranges, verbatim |
| `inputs.py` | builders: fire channel, ERA5-Land, VPD, FWI; FIRMS near-real-time fires for the newest days |
| `modis.py` | MODIS surface channels (in calibration) |
| `assemble.py` | an experiment root Niklas's `CubeReader` and `risk_map.py` read, from rebuilt and released modalities |
| `validate.py` | rebuilt vs released codes per band over Canadian land |
| `compare_maps.py` | risk maps from rebuilt vs released inputs, against fixed acceptance thresholds |

Earth Engine downloads are cached as `.npy` under `LIVE_CACHE`. Credentials are the Earth Engine client's own store
(`earthengine authenticate`).

## Recipe and fidelity per channel

Fidelity: share of Canadian land pixels whose rebuilt code equals the released one, on 2025 days.

| channel | recipe | fidelity |
|---|---|---|
| fire (1 band) | MOD14A1, **Terra only** (MPCFire's combine step paired Terra with itself: its `--myd_dir` default names the Terra folder), FireMask == 9, maximum over each release pixel's 5 x 5 block of 1 km pixels at the tile offsets below | every pixel, 2025-03-01, 07-13, 07-14, 07-15, 09-20 |
| ERA5 (18) | ERA5-Land hourly, daily over the 24 steps 00-23 UTC: means and maxima of the hourly values; the five sums as MPCFire's older DAILY_AGGR defined them, A(day 00:00) - A(previous day 23:00) + A(day 23:00) of the hourly accumulations (today's DAILY_AGGR ends at 00 UTC of the next day and misses MPCFire's sums by 1-3 codes); masked pixels as 0; GDAL bilinear onto the release grid | every pixel and band, 2025-03-01, 06-20, 07-14, 07-15, 10-25 |
| VPD (1) | Tetens from daily mean T2m and dewpoint on the 0.1-degree grid, values outside the valid ranges interpolated within 3 pixels (scipy griddata), bilinear | 99.9 % |
| FWI (6) | FireMPC's own recursion (its equations, deviations from Van Wagner included), every day of the year with state carried over, on the 0.1-degree grid; started from the released codes of a day | FFMC 41-47 %, DMC 45-73 %, DC 35-40 %, ISI 67-86 %, BUI 44-69 %, FWI 69-92 % exact and 70-97 % within one code over 2025-06-15 .. 10-25 with the earlier sums; to be re-measured with the exact sums |
| NDVI/EVI (2) | from the uint8 MCD09GA codes divided by 255 (MPCFire's route), then encoded | 98 % / 99.9 % exact from the released MCD09GA codes |
| MCD11A1 (4) | day-QC key chooses Terra or Aqua for all bands; LST_Day masked by day QC, emissivities by day or night QC; temporal fill from a 60-day spin-up; 10 x 10 window means at the measured offsets | 89 % / 75 % / 84 % / 82 % exact (one region, 10 days) |
| MCD09GA (4) | cloudy pixels dropped, QC key chooses the sensor, negative reflectance as no data, temporal fill, 10 x 10 windows | approximate: 4-15 % exact, mean 4-10 codes off (MPCFire's exact rule not found) |
| MCD09CMG (4) | Terra where valid else Aqua, temporal fill, bilinear from the 5600 m grid | not measured |
| MCD15A3H (2) | QA (SCF_QC <= 3), newest 4-day composite averaged onto each release pixel | not measured |

## Alignment

MPCFire downloaded Canada as 2 x 2 tiles and cut each tile into 5 km pixels from its own corner, then mosaicked
the tiles onto the north-west tile's grid by nearest neighbour. So a 1 km-based channel's release pixel (I, J)
holds 1 km pixels that start at (5I + dy, 5J + dx), with an offset per tile. Measured on released 2025 rasters:

| channel | seams (first south row, first east column) | north-west | north-east | south-west | south-east |
|---|---|---|---|---|---|
| fire: 5 x 5 blocks starting at (5I + dy, 5J + dx) | row 465, column 984; column 983 holds both tiles' blocks | (0, 0) | (0, -1) | (-2, 0) | (-2, -1) |
| MODIS land temperature: 10 x 10 windows starting at (5I + dy - 5, 5J + dx - 5) | assumed the same | (0, 0) | (0, -1) | (+2, 0) | (+2, -1) |

The fire offsets recover every released fire pixel of twelve 2025 days in each quarter; the seams come from every
fire-season day of 2025. The temperature offsets are the best of a 14 x 14 scan of window positions and of window
sizes 5 and 10 in each quarter.

## Acceptance

`compare_maps.py`, fixed on 2026-11-05 before any comparison was run: for the same date and model, risk maps from
rebuilt and from released inputs must agree at every lead time with Spearman >= 0.95 of log10 F, median |delta
log10 F| <= 0.10 and |delta AUC| <= 0.01 against the fires observed afterwards.

## Input lag (Earth Engine catalog, 2026-11-05)

ERA5-Land hourly through 2026-10-30 (DAILY_AGGR only through 10-26, which is why the hourly data is used), MOD14A1
through 10-31, MOD09GA, MOD11A1 and MOD09CMG through 11-03, MCD15A3H composites through 10-28. The newest forecast
date with every input is about five days old.

## Results (2026-11-05)

Risk maps from rebuilt vs released inputs (`compare_maps.py`):

| rebuilt | date | Spearman (tau 1, 7, 30) | median abs delta log10 F | delta AUC (7, 30) |
|---|---|---|---|---|
| fire, ERA5, VPD, FWI (MODIS released) | 2025-05-28 | 0.9996, 0.9999, 0.9999 | 0.000 | 0.0000, +0.0001 |
| fire, ERA5, VPD, FWI (MODIS released) | 2025-07-15 | 0.9998, 0.9999, 1.0000 | 0.000-0.001 | 0.0000, 0.0000 |
| everything | 2025-10-26 | 0.976, 0.984, 0.989 | 0.06-0.07 (p95 0.3) | -0.002, -0.002 |

The off-season reference (the model on MPCFire's own inputs of 2025-10-20 .. 26) ranks the fires that followed with
AUC 0.95-0.97 at 7 and 30 days, close to its in-season skill. The live forecasts of 2026-10-24 .. 30 rank the fires
observed since (5-11 days, through 2026-11-04) with AUC 0.96-0.97.
