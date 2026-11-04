/* Wildfire Survival Explorer: page logic.
 *
 * Reads the manifest embedded in #wf-data (schema wildfire-demo/1, written by tools/export_web.py) and
 * the basemap in #wf-basemap. Nothing here is specific to one region: the grid, dates, lead times and
 * models all come from the manifest. Rasters are drawn by two Leaflet GridLayers in plate carree
 * (EPSG:4326), the projection of the data grid, so no pixel is resampled across projections.
 */
(function () {
  'use strict';

  const manifest = JSON.parse(document.getElementById('wf-data').textContent);
  const basemap = JSON.parse(document.getElementById('wf-basemap').textContent);
  const G = manifest.grid;
  const Q = manifest.quant;
  const D = manifest.display;
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 9;

  // Light: matplotlib OrRd, the colormap of the paper's risk-map figure. Dark: the same hues with the
  // lightness reversed, so low risk recedes into the dark ground.
  const RAMPS = {
    light: ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000'],
    dark: ['#2b1d18', '#44201a', '#63231a', '#85281a', '#a8321b', '#c8441d', '#e15e25', '#f07f35', '#f7a14a'],
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs, text) => {
    const n = document.createElement(tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const svg = (tag, attrs, text) => {
    const n = document.createElementNS(SVG_NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  };

  // ---- values ---------------------------------------------------------------------------------------
  const LOG_QMIN = Math.log10(Q.vmin);
  const LOG_QSPAN = Math.log10(Q.vmax) - LOG_QMIN;
  const qToF = (q) => (q > 0 ? Math.pow(10, LOG_QMIN + ((q - 1) / (Q.levels - 1)) * LOG_QSPAN) : NaN);
  const LOG_DMIN = Math.log10(D.vmin);
  const LOG_DSPAN = Math.log10(D.vmax) - LOG_DMIN;
  const tOfF = (F) => Math.min(1, Math.max(0, (Math.log10(F) - LOG_DMIN) / LOG_DSPAN));

  function num(v) {
    if (v >= 10) return String(Math.round(v));
    return String(Number(v.toPrecision(2)));
  }
  function pct(F) {
    if (!(F > 0)) return '-';
    if (F <= Q.vmin * 1.001) return '< ' + num(Q.vmin * 100) + '%';
    return num(F * 100) + '%';
  }
  function fmtDate(iso, opts) {
    const d = new Date(iso + 'T00:00:00Z');
    return d.toLocaleDateString('en-US', Object.assign({ day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }, opts || {}));
  }
  function addDays(iso, n) {
    const d = new Date(iso + 'T00:00:00Z');
    d.setUTCDate(d.getUTCDate() + n);
    return d.toISOString().slice(0, 10);
  }
  function fmtLatLng(lat, lng) {
    return `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(2)}° ${lng >= 0 ? 'E' : 'W'}`;
  }
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

  // ---- colors ---------------------------------------------------------------------------------------
  const colors = { theme: 'light', lut: new Uint32Array(256), before: 0, after: 0, css: {} };
  const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');

  function effectiveTheme() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark' || t === 'light') return t;
    return darkQuery.matches ? 'dark' : 'light';
  }
  function hexToRgb(h) {
    h = h.trim().replace('#', '');
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    const v = parseInt(h, 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  const pack = (rgb) => ((255 << 24) | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0]) >>> 0;   // RGBA bytes, little-endian
  function rampColor(stops, t) {
    const x = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const a = stops[i], b = stops[i + 1];
    return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f));
  }
  function readTheme() {
    colors.theme = effectiveTheme();
    const cs = getComputedStyle(document.documentElement);
    for (const k of ['sea', 'land', 'coast', 'border', 'border-2', 'fire-before', 'fire-after', 'ink']) {
      colors.css[k] = cs.getPropertyValue('--' + k).trim();
    }
    const stops = RAMPS[colors.theme].map(hexToRgb);
    colors.stops = stops;
    colors.lut[0] = 0;
    for (let q = 1; q < 256; q++) colors.lut[q] = pack(rampColor(stops, tOfF(qToF(q))));
    colors.before = pack(hexToRgb(colors.css['fire-before']));
    colors.after = pack(hexToRgb(colors.css['fire-after']));
  }

  // ---- layers in memory -----------------------------------------------------------------------------
  // Embedded build: layer refs are data: URIs. A hosted build would resolve the same refs to URLs.
  function decodeLayer(ref) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const w = img.naturalWidth, h = img.naturalHeight;
        if (w !== G.width || h !== G.height) {
          reject(new Error(`a layer is ${w} × ${h} pixels, the grid is ${G.width} × ${G.height}`));
          return;
        }
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const rgba = ctx.getImageData(0, 0, w, h).data;
        const out = new Uint8Array(w * h);
        for (let i = 0, j = 0; i < out.length; i++, j += 4) out[i] = rgba[j];
        resolve(out);
      };
      img.onerror = () => reject(new Error('a map layer could not be decoded'));
      img.src = ref;
    });
  }

  // Pooled copies for zoomed-out views: level k covers 2^k × 2^k grid pixels per value, so small fires
  // and hot spots stay visible instead of falling between sampled pixels.
  const MAX_LEVEL = Math.max(0, Math.round(Math.log2(180 / (256 * Math.pow(2, MIN_ZOOM)) / G.res)));
  function pyramid(a, useMin) {
    const out = [a];
    let w = G.width, h = G.height, src = a;
    for (let k = 1; k <= MAX_LEVEL; k++) {
      const w2 = Math.ceil(w / 2), h2 = Math.ceil(h / 2), dst = new Uint8Array(w2 * h2);
      for (let y = 0; y < h2; y++) {
        for (let x = 0; x < w2; x++) {
          let m = useMin ? 255 : 0;
          for (let dy = 0; dy < 2; dy++) {
            const yy = 2 * y + dy;
            if (yy >= h) break;
            for (let dx = 0; dx < 2; dx++) {
              const xx = 2 * x + dx;
              if (xx >= w) break;
              const v = src[yy * w + xx];
              m = useMin ? Math.min(m, v) : Math.max(m, v);
            }
          }
          dst[y * w2 + x] = m;
        }
      }
      out.push(dst);
      w = w2; h = h2; src = dst;
    }
    return out;
  }

  const entries = new Map();   // "model|date" -> Promise of decoded layers; the two most recent are kept
  function forecastFor(model, date) {
    return manifest.forecasts.find((f) => f.model === model && f.date === date);
  }
  function loadEntry(fc) {
    const key = fc.model + '|' + fc.date;
    if (entries.has(key)) return entries.get(key);
    const p = (async () => {
      const risk = {};
      for (const t of fc.taus) {
        const raw = await decodeLayer(fc.risk[String(t)]);
        risk[t] = { raw, pyr: pyramid(raw, false) };
      }
      const obs = await decodeLayer(fc.observed);
      const win = new Uint8Array(obs.length), first = new Uint8Array(obs.length);
      for (let i = 0; i < obs.length; i++) {
        win[i] = obs[i] >> 7;
        const d = obs[i] & 127;
        first[i] = d === 0 ? 255 : d;
      }
      return { fc, risk, obs, winPyr: pyramid(win, false), firstPyr: pyramid(first, true) };
    })();
    entries.set(key, p);
    while (entries.size > 2) entries.delete(entries.keys().next().value);
    return p;
  }

  // ---- state ----------------------------------------------------------------------------------------
  const state = {
    model: manifest.models[0].id,
    date: null,
    tauIdx: 0,
    before: true,
    after: true,
    sel: null,          // {row, col}
    entry: null,        // decoded layers of the shown forecast
  };
  const datesFor = (model) => manifest.forecasts.filter((f) => f.model === model).map((f) => f.date).sort();
  const fcNow = () => forecastFor(state.model, state.date);
  const tauNow = () => fcNow().taus[state.tauIdx];

  // ---- map ------------------------------------------------------------------------------------------
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const map = L.map('map', {
    crs: L.CRS.EPSG4326, minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, zoomSnap: 0.25, zoomDelta: 0.5,
    attributionControl: false, zoomControl: false, worldCopyJump: false,
    maxBounds: [[-90, -180], [90, 180]], maxBoundsViscosity: 0.8,
    zoomAnimation: !reduceMotion, fadeAnimation: !reduceMotion,
  });
  L.control.zoom({ position: 'topright' }).addTo(map);
  L.control.scale({ imperial: false, position: 'bottomleft', maxWidth: 140 }).addTo(map);
  const gridBounds = L.latLngBounds(
    [G.lat0 - G.height * G.res, G.lon0], [G.lat0, G.lon0 + G.width * G.res]);
  [['wf-land', 200], ['wf-risk', 300], ['wf-lakes', 350], ['wf-fires', 380], ['wf-lines', 400], ['wf-select', 450]]
    .forEach(([name, z]) => { map.createPane(name).style.zIndex = z; });

  // basemap: delta-encoded integer lists -> Leaflet latlngs
  function decodeLine(a) {
    const s = basemap.scale, out = [];
    let x = 0, y = 0;
    for (let i = 0; i < a.length; i += 2) { x += a[i]; y += a[i + 1]; out.push([y / s, x / s]); }
    return out;
  }
  const renderer = (pane) => L.canvas({ pane, padding: 0.5 });
  const base = {
    land: L.polygon(basemap.land.map((p) => p.map(decodeLine)), { renderer: renderer('wf-land'), interactive: false, weight: 0.6, fillOpacity: 1 }),
    lakes: L.polygon(basemap.lakes.map((p) => p.map(decodeLine)), { renderer: renderer('wf-lakes'), interactive: false, weight: 0.6, fillOpacity: 1 }),
    provinces: L.polyline(basemap.provinces.map(decodeLine), { renderer: renderer('wf-lines'), interactive: false, weight: 0.6 }),
    borders: L.polyline(basemap.borders.map(decodeLine), { renderer: renderer('wf-lines'), interactive: false, weight: 1 }),
  };
  function styleBasemap() {
    const c = colors.css;
    base.land.setStyle({ color: c.coast, fillColor: c.land });
    base.lakes.setStyle({ color: c.coast, fillColor: c.sea });
    base.provinces.setStyle({ color: c['border-2'] });
    base.borders.setStyle({ color: c.border });
  }

  // rasters
  const levelFor = (z) => Math.min(MAX_LEVEL, Math.max(0, Math.round(Math.log2(180 / (256 * Math.pow(2, z)) / G.res))));
  function tileIndex(coords, level) {
    const scale = 256 * Math.pow(2, coords.z);          // EPSG:4326 in Leaflet: 256 * 2^z pixels per 180 degrees
    const f = 1 << level;
    const w = Math.ceil(G.width / f), h = Math.ceil(G.height / f);
    const cols = new Int32Array(256), rows = new Int32Array(256);
    for (let i = 0; i < 256; i++) {
      const lng = ((coords.x * 256 + i + 0.5) / scale - 1) * 180;
      const c = Math.floor((lng - G.lon0) / G.res / f);
      cols[i] = c >= 0 && c < w ? c : -1;
    }
    for (let j = 0; j < 256; j++) {
      const lat = (0.5 - (coords.y * 256 + j + 0.5) / scale) * 180;
      const r = Math.floor((G.lat0 - lat) / G.res / f);
      rows[j] = r >= 0 && r < h ? r : -1;
    }
    return { cols, rows, w };
  }
  function paintRisk(canvas, coords) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 256);
    const e = state.entry;
    if (!e) return;
    const level = levelFor(coords.z);
    const a = e.risk[tauNow()].pyr[level];
    const { cols, rows, w } = tileIndex(coords, level);
    const img = ctx.createImageData(256, 256), px = new Uint32Array(img.data.buffer), lut = colors.lut;
    for (let j = 0; j < 256; j++) {
      const r = rows[j];
      if (r < 0) continue;
      const rowBase = r * w, o = j * 256;
      for (let i = 0; i < 256; i++) {
        const c = cols[i];
        if (c < 0) continue;
        const v = a[rowBase + c];
        if (v) px[o + i] = lut[v];
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  function paintFires(canvas, coords) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 256);
    const e = state.entry;
    if (!e || (!state.before && !state.after)) return;
    const level = levelFor(coords.z);
    const win = e.winPyr[level], first = e.firstPyr[level], tau = tauNow();
    const area = e.risk[tau].pyr[level];   // fires are drawn only where a forecast exists
    const { cols, rows, w } = tileIndex(coords, level);
    const img = ctx.createImageData(256, 256), px = new Uint32Array(img.data.buffer);
    for (let j = 0; j < 256; j++) {
      const r = rows[j];
      if (r < 0) continue;
      const rowBase = r * w, o = j * 256;
      for (let i = 0; i < 256; i++) {
        const c = cols[i];
        if (c < 0) continue;
        const k = rowBase + c;
        if (!area[k]) continue;
        if (state.after && first[k] <= tau) px[o + i] = colors.after;          // after the date, drawn on top
        else if (state.before && win[k]) px[o + i] = colors.before;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  const RasterLayer = L.GridLayer.extend({
    createTile(coords) {
      const t = L.DomUtil.create('canvas');
      t.width = 256; t.height = 256;
      this.options.paint(t, coords);
      return t;
    },
    repaint() {   // redraw the tiles in place, without the flash of recreating them
      for (const key in this._tiles) this.options.paint(this._tiles[key].el, this._tiles[key].coords);
    },
  });
  const rasterOpts = { bounds: gridBounds, noWrap: true, updateWhenZooming: false, keepBuffer: 2, className: 'wf-raster' };
  const riskLayer = new RasterLayer(Object.assign({ pane: 'wf-risk', paint: paintRisk }, rasterOpts));
  const fireLayer = new RasterLayer(Object.assign({ pane: 'wf-fires', paint: paintFires }, rasterOpts));

  base.land.addTo(map);
  riskLayer.addTo(map);
  base.lakes.addTo(map);
  fireLayer.addTo(map);
  base.provinces.addTo(map);
  base.borders.addTo(map);
  const selBox = L.rectangle(gridBounds, {
    renderer: renderer('wf-select'), interactive: false, fill: false, weight: 2, opacity: 0,
  }).addTo(map);
  map.fitBounds(gridBounds, { padding: [24, 24] });

  // ---- reading the map ------------------------------------------------------------------------------
  function cellAt(latlng) {
    const col = Math.floor((latlng.lng - G.lon0) / G.res);
    const row = Math.floor((G.lat0 - latlng.lat) / G.res);
    if (col < 0 || col >= G.width || row < 0 || row >= G.height) return null;
    return { row, col };
  }
  function cellCenter(cell) {
    return [G.lat0 - (cell.row + 0.5) * G.res, G.lon0 + (cell.col + 0.5) * G.res];
  }
  function observedText(e, idx, long) {
    const o = e.obs[idx], d = o & 127, burning = (o >> 7) === 1, n = e.fc.observed_days;
    const parts = [];
    if (burning) parts.push(long ? `Fire detected in the ${manifest.input_days} days before the forecast date.` : 'Fire in the input window');
    if (d > 0) parts.push(long ? `First fire after the forecast date: day ${d}.` : `First fire after the date: day ${d}`);
    else parts.push(long ? `No fire detected within ${n} days after the forecast date.` : `No fire in the ${n} days after the date`);
    return parts;
  }

  const tip = $('tip');
  map.on('mousemove', (ev) => {
    const e = state.entry, cell = cellAt(ev.latlng);
    if (!e || !cell) { tip.hidden = true; return; }
    const idx = cell.row * G.width + cell.col;
    const q = e.risk[tauNow()].raw[idx];
    tip.replaceChildren();
    tip.append(el('div', { class: 'muted' }, fmtLatLng(ev.latlng.lat, ev.latlng.lng)));
    if (q) {
      const line = el('div');
      line.append(document.createTextNode(`Fire within ${plural(tauNow(), 'day')}: `), el('b', {}, pct(qToF(q))));
      tip.append(line);
      for (const t of observedText(e, idx, false)) tip.append(el('div', {}, t));
    } else {
      tip.append(el('div', {}, 'Outside the forecast area'));
    }
    const p = ev.containerPoint, box = map.getSize();
    tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(p.x + 14, box.x - tw - 8) + 'px';
    tip.style.top = (p.y + 14 + th > box.y ? p.y - th - 10 : p.y + 14) + 'px';
  });
  map.on('mouseout', () => { tip.hidden = true; });
  map.on('click', (ev) => {
    const cell = cellAt(ev.latlng);
    if (!cell || !state.entry || !state.entry.risk[tauNow()].raw[cell.row * G.width + cell.col]) return;
    state.sel = cell;
    renderSelection();
  });

  // ---- panel ----------------------------------------------------------------------------------------
  function renderDates() {
    const box = $('dates');
    box.replaceChildren();
    for (const d of datesFor(state.model)) {
      const b = el('button', { type: 'button', class: 'date-btn', 'aria-pressed': String(d === state.date) }, fmtDate(d));
      b.addEventListener('click', () => setDate(d));
      box.append(b);
    }
    const fc = fcNow();
    $('window-text').textContent =
      `Model input: daily maps from ${fmtDate(addDays(fc.date, 1 - manifest.input_days), { year: undefined })} to ${fmtDate(fc.date)} (${manifest.input_days} days).`;
  }
  function renderLead() {
    const fc = fcNow(), tau = tauNow(), slider = $('lead');
    slider.max = String(fc.taus.length - 1);
    slider.value = String(state.tauIdx);
    slider.setAttribute('aria-valuetext', plural(tau, 'day'));
    $('lead-num').textContent = String(tau);
    $('lead-unit').textContent = tau === 1 ? 'day' : 'days';
    $('lead-text').textContent =
      `Probability of a first fire within ${plural(tau, 'day')} after ${fmtDate(fc.date)}, that is by ${fmtDate(addDays(fc.date, tau))}.`;
    const ticks = $('ticks');
    ticks.replaceChildren();
    const n = fc.taus.length;
    fc.taus.forEach((t, i) => {
      const s = el('span', {}, String(t));
      s.style.left = n > 1 ? `calc(8px + (100% - 16px) * ${i / (n - 1)})` : '50%';
      ticks.append(s);
    });
  }
  function renderTitle() {
    const t = $('map-title');
    t.replaceChildren(document.createTextNode(`Risk of a fire within ${plural(tauNow(), 'day')}`),
      el('span', { class: 'muted' }, ` · forecast date ${fmtDate(state.date)}`));
    $('legend-title').textContent = `Probability of a first fire in the pixel within ${plural(tauNow(), 'day')}`;
  }
  function renderKey() {
    const m = manifest.models.find((x) => x.id === state.model);
    const mt = $('model-text');
    mt.replaceChildren(el('b', {}, m.label), document.createTextNode((m.detail ? ', ' + m.detail : '') + '.'));
    const css = colors.stops.map((c, i) => `rgb(${c.join(' ')}) ${(100 * i) / (colors.stops.length - 1)}%`).join(', ');
    $('ramp').style.background = `linear-gradient(to right, ${css})`;
    const ticks = $('ramp-ticks');
    ticks.replaceChildren();
    const marks = [D.vmin];
    for (let e = Math.ceil(LOG_DMIN + 1e-9); Math.pow(10, e) < D.vmax * 0.6; e++) if (Math.pow(10, e) > D.vmin) marks.push(Math.pow(10, e));
    marks.push(D.vmax);
    for (const F of marks) {
      const s = el('span', {}, num(F * 100) + '%');
      s.style.left = (100 * tOfF(F)) + '%';
      ticks.append(s);
    }
  }

  // risk curve of the selected pixel
  function renderSelection() {
    const e = state.entry;
    if (!e || !state.sel) return;
    const fc = e.fc, idx = state.sel.row * G.width + state.sel.col;
    const [lat, lng] = cellCenter(state.sel);
    $('loc-name').textContent = fmtLatLng(lat, lng);
    const h2 = G.res / 2;
    selBox.setBounds([[lat - h2, lng - h2], [lat + h2, lng + h2]]);
    selBox.setStyle({ opacity: 1, color: colors.css.ink });

    const pts = fc.taus.map((t) => ({ t, F: qToF(e.risk[t].raw[idx]) }));
    const W = 320, H = 178, m = { l: 46, r: 10, t: 16, b: 28 };
    const tMax = fc.taus[fc.taus.length - 1];
    const yLo = Math.log10(Q.vmin), yHi = 0;
    const x = (t) => m.l + (t / tMax) * (W - m.l - m.r);
    const y = (F) => m.t + (1 - (Math.log10(Math.max(F, Q.vmin)) - yLo) / (yHi - yLo)) * (H - m.t - m.b);
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Risk curve of the selected location' });
    const grid = svg('g', { class: 'grid' }), axis = svg('g', { class: 'axis' });
    for (let p = Math.ceil(yLo); p <= yHi; p++) {
      const yy = y(Math.pow(10, p));
      grid.append(svg('line', { x1: m.l, x2: W - m.r, y1: yy, y2: yy }));
      axis.append(svg('text', { x: m.l - 6, y: yy + 3.5, 'text-anchor': 'end' }, num(Math.pow(10, p) * 100) + '%'));
    }
    for (const t of fc.taus) {
      axis.append(svg('text', { x: x(t), y: H - m.b + 15, 'text-anchor': 'middle' }, String(t)));
    }
    axis.append(svg('text', { x: W - m.r, y: H - 2, 'text-anchor': 'end' }, 'days after the forecast date'));
    s.append(grid, axis);
    const firstDay = e.obs[idx] & 127;
    if (firstDay > 0) {
      const g = svg('g', { class: 'obs' });
      const day = Math.min(firstDay, tMax), xx = x(day);
      g.append(svg('line', { x1: xx, x2: xx, y1: m.t, y2: H - m.b }));
      // label where the curve is not: next to the axis when the curve runs high at that day
      const near = pts.reduce((a, p) => (Math.abs(p.t - day) < Math.abs(a.t - day) ? p : a));
      const low = y(near.F) < (m.t + H - m.b) / 2;
      const right = xx > W - 110;
      g.append(svg('text', { x: right ? xx - 4 : xx + 4, y: low ? H - m.b - 5 : m.t + 9, 'text-anchor': right ? 'end' : 'start' }, `fire on day ${firstDay}`));
      s.append(g);
    }
    s.append(svg('polyline', { class: 'line', points: pts.map((p) => `${x(p.t)},${y(p.F)}`).join(' ') }));
    pts.forEach((p, i) => {
      const c = svg('circle', { class: 'pt' + (i === state.tauIdx ? ' on' : ''), cx: x(p.t), cy: y(p.F), r: i === state.tauIdx ? 4.5 : 3.5 });
      c.append(svg('title', {}, `${plural(p.t, 'day')}: ${pct(p.F)}`));
      s.append(c);
    });
    $('curve').replaceChildren(s);

    $('loc-outcome').textContent = 'Observed: ' + observedText(e, idx, true).join(' ');
    const table = $('vals');
    table.replaceChildren();
    const head = el('tr');
    head.append(el('th', { scope: 'col' }, 'Within'), el('th', { scope: 'col' }, 'Probability'));
    table.append(head);
    pts.forEach((p, i) => {
      const tr = el('tr', i === state.tauIdx ? { class: 'on' } : {});
      tr.append(el('td', {}, plural(p.t, 'day')), el('td', {}, pct(p.F)));
      table.append(tr);
    });
  }

  // A starting location that shows what a risk curve is for: below 1% for the first lead time, the highest
  // risk at the last one, no fire in the input window, and a fire observed afterwards. It is an example;
  // the reader can click anywhere. Without such a pixel: the highest risk at the shown lead time.
  const fToQ = (F) => 1 + Math.round(((Math.log10(F) - LOG_QMIN) / LOG_QSPAN) * (Q.levels - 1));
  function pickStart(e) {
    const taus = e.fc.taus, first = e.risk[taus[0]].raw, last = e.risk[taus[taus.length - 1]].raw;
    const cap = fToQ(0.01);
    let best = -1, bestQ = 0;
    for (let i = 0; i < last.length; i++) {
      const o = e.obs[i];
      if ((o >> 7) === 0 && (o & 127) > 0 && first[i] > 0 && first[i] < cap && last[i] > bestQ) { bestQ = last[i]; best = i; }
    }
    if (best < 0) {
      const raw = e.risk[tauNow()].raw;
      for (let i = 0; i < raw.length; i++) if (raw[i] > bestQ && (e.obs[i] >> 7) === 0) { bestQ = raw[i]; best = i; }
    }
    return best < 0 ? null : { row: Math.floor(best / G.width), col: best % G.width };
  }

  // ---- updates --------------------------------------------------------------------------------------
  function repaintAll() {
    riskLayer.repaint();
    fireLayer.repaint();
  }
  function showError(err) {
    const n = $('error-note');
    n.textContent = `This page could not show the forecast: ${err.message}.`;
    n.hidden = false;
    window.__wfReady = true;
  }
  async function setDate(date) {
    state.date = date;
    const fc = fcNow();
    state.tauIdx = Math.min(state.tauIdx, fc.taus.length - 1);
    renderDates(); renderLead(); renderTitle();
    try {
      const e = await loadEntry(fc);
      if (state.date !== date) return;
      state.entry = e;
      if (!state.sel) state.sel = pickStart(e);
      repaintAll();
      renderSelection();
    } catch (err) { showError(err); }
  }
  function setTau(i) {
    state.tauIdx = i;
    renderLead(); renderTitle(); repaintAll(); renderSelection();
  }
  function applyTheme() {
    readTheme();
    styleBasemap();
    renderKey();
    repaintAll();
    renderSelection();
  }

  $('lead').addEventListener('input', (ev) => setTau(Number(ev.target.value)));
  $('show-before').addEventListener('change', (ev) => { state.before = ev.target.checked; fireLayer.repaint(); });
  $('show-after').addEventListener('change', (ev) => { state.after = ev.target.checked; fireLayer.repaint(); });
  darkQuery.addEventListener('change', applyTheme);
  new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ---- start ----------------------------------------------------------------------------------------
  $('placeholder-note').hidden = !manifest.placeholder;
  const notes = $('notes');
  for (const t of manifest.notes || []) notes.append(el('li', {}, t));
  readTheme();
  styleBasemap();
  renderKey();
  // opening view: the manifest's default (the paper figure's date and lead time), else the first date at 7 days
  const dates = datesFor(state.model);
  const want = manifest.default || {};
  const startDate = dates.includes(want.date) ? want.date : dates[0];
  const startFc = forecastFor(state.model, startDate);
  const it = startFc.taus.indexOf(want.tau || 7);
  state.tauIdx = it >= 0 ? it : Math.floor(startFc.taus.length / 2);
  setDate(startDate).then(() => {
    requestAnimationFrame(() => { window.__wfReady = true; });
  });
})();
