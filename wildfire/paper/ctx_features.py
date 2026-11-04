"""Zusatzeingaben fuer Extremlagen (Reviewer-Kritik, Punkte 2 und 3): Saisonzustand der Region und
Einordnung des Feuerwetters relativ zur eigenen Zelle. Nur Vergangenheit bis einschliesslich Stichtag.

  Pixel (je Stichtag 14 x 13 x 13, Werte am Stichtag aus dem Eingabefenster)
    pct_<k>    Perzentil (Mid-Rank) des Werts in der Verteilung DERSELBEN Zelle 2000-2019 (alle
               Saisontage), 0..1, fuer FFMC, DMC, DC, ISI, BUI, FWI
    above_<k>  Wert ueber dem Maximum derselben Zelle 2000-2019, fuer DC, BUI, FWI
    sat_<k>    Wert 255 in einer Zelle, die nicht NoData ist (Kappung des uint8), DC, BUI, FWI, ISI
    nodata     FWI in den Trainingsjahren an >= 99 % der Zelltage 255: kein Wert, nicht extrem
  Region (je Stichtag 8 Skalare): MODIS-Feuer-Pixeltage aus dem Feuerkanal des Wuerfels, summiert in
  Bloecken von 16 x 16 Pixeln
    n30_r3, n30_r8   letzte 30 Tage, Bloecke im Chebyshev-Radius 3 bzw. 8 um den Block des Stichtags
    n60_r8           letzte 60 Tage (nur Saisontage), Radius 8
    ytd_r8, ytd_ca   seit Saisonbeginn, Radius 8 bzw. ganz Kanada
    an30_r8, anytd_r8, anytd_ca   Anomalie log((x + 1) / (Klima + 1)), Klima = Mittel 2000-2019
                     desselben Saisontags (und Blocks)
    Zaehlungen als log1p.

Ein Lauf liest den Wuerfel einmal in Zeilenbaendern (alle Jahre: Feuerbloecke; Trainingsjahre:
Histogramme je Zelle) und schreibt nach derived_data/ctx_features/ (NEU):
  cell_pct_lut.npy (H, W, 6, 256) uint8   Perzentil * 255 je Zelle, Kanal und uint8-Wert
  cell_max.npy (H, W, 6) uint8, nodata.npy (H, W) bool, share255_fwi.npy (H, W) float16
  <menge>_px.npy (n, 14, 13, 13) float16 und <menge>_reg.npy (n, 8) float32 fuer
    train          derived_data/paper_pixel/train_meta.npz (alle Zeilen, Reihenfolge wie dort)
    surv_val, surv_eval   derived_data/surv_eval_test/{val,eval}_meta.npz
    cache          derived_data/paper/cache_G_L61, Stichtag e = D - dt je dt in compare.DTS:
                   (N, 4, 14, 13, 13) bzw. (N, 4, 8); Tage ausserhalb der Wuerfel-Saison nehmen fuer die
                   Region den naechsten Saisontag desselben Jahres
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import time
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.compare import DTS

PCT = {"FFMC": 19, "DMC": 20, "DC": 21, "ISI": 22, "BUI": 23, "FWI": 24}   # Wuerfel = Cache-Index
ABOVE = ("DC", "BUI", "FWI")
SAT = ("DC", "BUI", "FWI", "ISI")
PX_NAMES = [f"pct_{k}" for k in PCT] + [f"above_{k}" for k in ABOVE] + [f"sat_{k}" for k in SAT] + ["nodata"]
REG_NAMES = ["n30_r3", "n30_r8", "n60_r8", "ytd_r8", "ytd_ca", "an30_r8", "anytd_r8", "anytd_ca"]
TRAIN_YEARS = (2000, 2019)
BLK = 16
BAND = 64
HALF = 6
FIRE_CH = 0


def box(a: np.ndarray, R: int) -> np.ndarray:
    """(T, nr, nc) -> Summe ueber die Bloecke im Chebyshev-Radius R (Integralbild)."""
    T, nr, nc = a.shape
    ii = np.zeros((T, nr + 1, nc + 1), np.float64)
    ii[:, 1:, 1:] = a.cumsum(1).cumsum(2)
    i0, i1 = np.clip(np.arange(nr) - R, 0, nr), np.clip(np.arange(nr) + R + 1, 0, nr)
    j0, j1 = np.clip(np.arange(nc) - R, 0, nc), np.clip(np.arange(nc) + R + 1, 0, nc)
    g = lambda i, j: ii[:, i][:, :, j]  # noqa: E731
    return g(i1, j1) - g(i0, j1) - g(i1, j0) + g(i0, j0)


def last_k(a: np.ndarray, k: int) -> np.ndarray:
    """Summe ueber die letzten k Tage einschliesslich t (am Saisonbeginn verkuerzt), Achse 0."""
    cs = np.concatenate([np.zeros_like(a[:1], dtype=np.float64), a.cumsum(0, dtype=np.float64)])
    t = np.arange(a.shape[0])
    return cs[t + 1] - cs[np.maximum(t - k + 1, 0)]


def scan_cube(cube: CubeReader, years: list[str]):
    """Ein Durchlauf in Zeilenbaendern: Feuer-Pixeltage je Block und Tag (alle Jahre), Histogramme je
    Zelle fuer die PCT-Kanaele (Trainingsjahre) -> LUT, Zellmaximum, NoData."""
    H, W = cube.H, cube.W
    nbr, nbc = H // BLK, W // BLK
    assert H % BLK == 0 and W % BLK == 0 and BAND % BLK == 0, (H, W)
    chs = list(PCT.values())
    T = {y: int(cube.per_year[y]["dyn_shape"][0]) for y in years}
    fire = {y: np.zeros((T[y], nbr, nbc), np.int32) for y in years}
    lut = np.zeros((H, W, len(chs), 256), np.uint8)
    cmax = np.zeros((H, W, len(chs)), np.uint8)
    s255 = np.zeros((H, W), np.float32)
    train = [y for y in years if TRAIN_YEARS[0] <= int(y) <= TRAIN_YEARS[1]]
    t0 = time.time()
    for r0 in range(0, H, BAND):
        r1 = min(H, r0 + BAND)
        nc = (r1 - r0) * W
        hist = np.zeros((len(chs), nc * 256), np.int64)
        cell = (np.arange(nc, dtype=np.int64) * 256)[None]
        for y in years:
            dyn, _ = cube._arrays(y)
            a = np.asarray(dyn[:, r0:r1])                                   # (T, rows, W, 42)
            f = (a[..., FIRE_CH] > 0).reshape(T[y], (r1 - r0) // BLK, BLK, nbc, BLK).sum((2, 4))
            fire[y][:, r0 // BLK:r1 // BLK] = f
            if y in train:
                for j, ch in enumerate(chs):
                    v = a[..., ch].reshape(T[y], nc).astype(np.int64)
                    hist[j] += np.bincount((cell + v).ravel(), minlength=nc * 256)
            del a
        h = hist.reshape(len(chs), nc, 256).transpose(1, 0, 2).astype(np.float64)   # (nc, 6, 256)
        n = h.sum(-1, keepdims=True)
        below = np.cumsum(h, -1) - h
        pct = (below + 0.5 * h) / np.maximum(n, 1)
        lut[r0:r1] = np.round(255 * pct).astype(np.uint8).reshape(r1 - r0, W, len(chs), 256)
        seen = h > 0
        cmax[r0:r1] = (255 - np.argmax(seen[..., ::-1], -1)).reshape(r1 - r0, W, len(chs))
        s255[r0:r1] = (h[:, chs.index(PCT["FWI"]), 255] / np.maximum(n[:, chs.index(PCT["FWI"]), 0], 1)).reshape(r1 - r0, W)
        print(f"[ctx] Zeilen {r0}-{r1} fertig ({time.time() - t0:.0f} s)", flush=True)
    return fire, lut, cmax, s255, T


def regional_grids(fire: dict, T: dict, years: list[str]) -> dict:
    """Je Jahr (T, nbr, nbc, 8) float32 in der Reihenfolge REG_NAMES."""
    raw = {}
    for y in years:
        a = fire[y].astype(np.float64)
        b8 = box(a, 8)
        raw[y] = {"n30_r3": last_k(box(a, 3), 30), "n30_r8": last_k(b8, 30), "n60_r8": last_k(b8, 60),
                  "ytd_r8": b8.cumsum(0), "ytd_ca": np.broadcast_to(a.sum((1, 2)).cumsum(0)[:, None, None], b8.shape)}
    train = [y for y in years if TRAIN_YEARS[0] <= int(y) <= TRAIN_YEARS[1]]
    if len({T[y] for y in years}) != 1:
        raise SystemExit(f"ABBRUCH: Jahre mit unterschiedlich vielen Saisontagen {T}")
    clim = {k: np.mean([raw[y][k] for y in train], 0) for k in ("n30_r8", "ytd_r8", "ytd_ca")}
    out = {}
    for y in years:
        r = raw[y]
        feats = [np.log1p(r[k]) for k in REG_NAMES[:5]]
        feats += [np.log((r[k] + 1) / (clim[k] + 1)) for k in ("n30_r8", "ytd_r8", "ytd_ca")]
        out[y] = np.stack(feats, -1).astype(np.float32)
    return out


def season_first(cube: CubeReader, y: str) -> dt.date:
    days = cube.per_year[y]["days"]
    days = days if isinstance(days, list) else ast.literal_eval(days)
    return dt.date.fromisoformat(str(days[0])[:10])


def season_pos(cube: CubeReader, date: str) -> tuple[str, int]:
    """ISO-Datum -> (Jahr, Saisontag), ausserhalb der Saison auf den naechsten Saisontag desselben Jahres."""
    y = date[:4]
    first = season_first(cube, y)
    li = (dt.date.fromisoformat(date) - first).days
    return y, int(np.clip(li, 0, int(cube.per_year[y]["dyn_shape"][0]) - 1))


def px_features(V: np.ndarray, rows: np.ndarray, cols: np.ndarray, lut, cmax, nodata, H: int, W: int) -> np.ndarray:
    """V (n, 6, 13, 13) uint8 Werte am Stichtag (PCT-Reihenfolge), Mittelzelle rows/cols -> (n, 14, 13, 13)."""
    n = V.shape[0]
    dr = np.arange(-HALF, HALF + 1)
    R = rows[:, None, None] + dr[None, :, None]
    C = cols[:, None, None] + dr[None, None, :]
    ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)                            # (n, 13, 13)
    Rc, Cc = np.clip(R, 0, H - 1), np.clip(C, 0, W - 1)
    out = np.zeros((n, len(PX_NAMES), 2 * HALF + 1, 2 * HALF + 1), np.float32)
    nd = nodata[Rc, Cc] & ok
    names = list(PCT)
    for j in range(len(names)):
        v = V[:, j].astype(np.int64)
        out[:, j] = lut[Rc, Cc, j, v] / 255.0
    for m, k in enumerate(ABOVE):
        j = names.index(k)
        out[:, len(names) + m] = V[:, j] > cmax[Rc, Cc, j]
    for m, k in enumerate(SAT):
        j = names.index(k)
        out[:, len(names) + len(ABOVE) + m] = (V[:, j] == 255) & ~nd
    out[:, -1] = nd
    return out * ok[:, None]


def landmark_values(X, sel: np.ndarray, step: int | None, chunk: int = 4096) -> np.ndarray:
    """Werte der PCT-Kanaele am Stichtag aus einem Fenster-Array (n, T, 55, 13, 13), Zeilen sel
    (aufsteigend): Fenstertag -1 bzw. step."""
    chs = list(PCT.values())
    out = np.zeros((sel.size, len(chs), 2 * HALF + 1, 2 * HALF + 1), np.uint8)
    for a in range(0, sel.size, chunk):
        s = sel[a:a + chunk]
        if s.size and s[-1] - s[0] == s.size - 1:                        # zusammenhaengend: Slice statt Fancy
            day = np.asarray(X[s[0]:s[-1] + 1, -1 if step is None else step])
        else:
            day = np.asarray(X[s][:, -1 if step is None else step])
        out[a:a + chunk] = day[:, chs]
    return out


def self_test_lut(cube: CubeReader, lut, years: list[str], n: int = 40, seed: int = 0) -> float:
    """Perzentil aus der LUT gegen direkt gelesene Zellreihen (Trainingsjahre) fuer zufaellige Zellen."""
    rng = np.random.default_rng(seed)
    rr, cc = rng.integers(0, cube.H, n), rng.integers(0, cube.W, n)
    train = [y for y in years if TRAIN_YEARS[0] <= int(y) <= TRAIN_YEARS[1]]
    ch = PCT["DC"]
    err = []
    for r, c in zip(rr, cc):
        s = np.concatenate([np.asarray(cube._arrays(y)[0][:, r, c, ch]) for y in train]).astype(np.int64)
        for v in np.unique(s)[:20]:
            ref = ((s < v).sum() + 0.5 * (s == v).sum()) / s.size
            err.append(abs(lut[r, c, list(PCT).index("DC"), v] / 255.0 - ref))
    return float(np.max(err))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--out-name", default="ctx_features")
    ap.add_argument("--years", nargs="*", default=None, help="nur fuer Rauchtests: diese Wuerfeljahre")
    ap.add_argument("--limit", type=int, default=0, help="nur fuer Rauchtests: so viele Stichtage je Menge")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "derived_data" / args.out_name
    if (out / "cell_pct_lut.npy").exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --out-name waehlen")
    out.mkdir(parents=True, exist_ok=True)
    cube = CubeReader(exp)
    years = sorted(cube.per_year) if not args.years else sorted(args.years)
    t0 = time.time()
    fire, lut, cmax, s255, T = scan_cube(cube, years)
    nodata = s255 >= 0.99
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    print(f"[ctx] Wuerfel gelesen in {time.time() - t0:.0f} s; NoData-Anteil der Landzellen {nodata[land].mean():.4f}",
          flush=True)
    np.save(out / "cell_pct_lut.npy", lut)
    np.save(out / "cell_max.npy", cmax)
    np.save(out / "nodata.npy", nodata)
    np.save(out / "share255_fwi.npy", s255.astype(np.float16))
    lut_err = self_test_lut(cube, lut, years)
    print(f"[ctx] Selbsttest LUT gegen direkte Zellreihen (DC): max. Abweichung {lut_err:.4f}", flush=True)
    if lut_err > 1.0 / 255 + 1e-6:
        raise SystemExit("ABBRUCH: LUT passt nicht zu den Zellreihen")
    reg = regional_grids(fire, T, years)
    season_end = {y: {k: float(reg[y][-1, 0, 0, REG_NAMES.index(k)]) for k in ("ytd_ca", "anytd_ca")} for y in years}
    for y in years:
        print(f"[ctx] {y}: Kanada Feuer-Pixeltage Saison log1p {season_end[y]['ytd_ca']:.2f}, "
              f"Anomalie {season_end[y]['anytd_ca']:+.2f}", flush=True)

    def reg_at(yl: list[tuple[str, int]], rows, cols) -> np.ndarray:
        return np.stack([reg[y][li, r // BLK, c // BLK] for (y, li), r, c in zip(yl, rows, cols)]).astype(np.float32)

    sets = {}

    def pick(yrs: np.ndarray, name: str) -> np.ndarray:
        """Zeilen, deren Jahr im Lauf liegt; im vollen Lauf muessen es alle sein."""
        sel = np.where(np.isin(yrs, years))[0]
        if not args.years and sel.size < yrs.size:
            raise SystemExit(f"ABBRUCH: {name}: {yrs.size - sel.size} Stichtage ausserhalb der Wuerfeljahre")
        return sel[:args.limit] if args.limit else sel
    # ---- Training (Wuerfel-Stichtage) und surv_eval-Stichtage
    for name, mpath, xpath, lkey in (
            ("train", exp / "derived_data" / args.cube_dir / "train_meta.npz",
             exp / "derived_data" / args.cube_dir / "train_X.npy", "L"),
            ("surv_val", exp / "derived_data" / args.surv_dir / "val_meta.npz",
             exp / "derived_data" / args.surv_dir / "val_X.npy", "L_cube"),
            ("surv_eval", exp / "derived_data" / args.surv_dir / "eval_meta.npz",
             exp / "derived_data" / args.surv_dir / "eval_X.npy", "L_cube")):
        t1 = time.time()
        m = np.load(mpath)
        sel = pick(np.array([cube._pos[int(L)][0] for L in m[lkey]]), name)
        if sel.size == 0:
            print(f"[ctx] {name}: keine Stichtage in den Jahren des Laufs, uebersprungen", flush=True)
            continue
        rows, cols, Ls = m["row"][sel].astype(np.int64), m["col"][sel].astype(np.int64), m[lkey][sel].astype(np.int64)
        V = landmark_values(np.load(xpath, mmap_mode="r"), sel, None)
        px = px_features(V, rows, cols, lut, cmax, nodata, cube.H, cube.W)
        rg = reg_at([cube._pos[int(L)] for L in Ls], rows, cols)
        np.save(out / f"{name}_px.npy", px.astype(np.float16))
        np.save(out / f"{name}_reg.npy", rg)
        if args.years:
            np.save(out / f"{name}_rows.npy", sel)
        sets[name] = {"n": int(rows.size), "px_mean": dict(zip(PX_NAMES, px.mean((0, 2, 3)).round(4).tolist())),
                      "reg_mean": dict(zip(REG_NAMES, rg.mean(0).round(4).tolist()))}
        print(f"[ctx] {name}: {rows.size:,} Stichtage in {time.time() - t1:.0f} s; "
              + ", ".join(f"{k} {v:.3f}" for k, v in sets[name]["px_mean"].items()), flush=True)

    # ---- Paper-Cache: Stichtag e = D - dt (Index LC - 1 - dt)
    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    meta = np.load(cd / "meta.npy")
    X = np.load(cd / "X.npy", mmap_mode="r")
    LC = X.shape[1]
    sel = pick(np.array([str(int(v) // 10000) for v in meta[:, 0]]), "cache")
    N = sel.size
    rows, cols = meta[sel, 1].astype(np.int64), meta[sel, 2].astype(np.int64)
    Dd = [dt.date(int(v) // 10000, int(v) // 100 % 100, int(v) % 100) for v in meta[sel, 0]]
    if N:
        px_c = np.zeros((N, len(DTS), len(PX_NAMES), 2 * HALF + 1, 2 * HALF + 1), np.float16)
        rg_c = np.zeros((N, len(DTS), len(REG_NAMES)), np.float32)
        outside = 0
        for k, d_ in enumerate(DTS):
            V = landmark_values(X, sel, LC - 1 - d_)
            px_c[:, k] = px_features(V, rows, cols, lut, cmax, nodata, cube.H, cube.W)
            pos = []
            for D in Dd:
                e = (D - dt.timedelta(days=d_)).isoformat()
                y, li = season_pos(cube, e)
                first = season_first(cube, y)
                outside += int((first + dt.timedelta(days=li)).isoformat() != e)
                pos.append((y, li))
            rg_c[:, k] = reg_at(pos, rows, cols)
        np.save(out / "cache_px.npy", px_c)
        np.save(out / "cache_reg.npy", rg_c)
        if args.years:
            np.save(out / "cache_rows.npy", sel)
        sets["cache"] = {"n": int(N), "dts": list(DTS), "region_day_clamped": outside}
        print(f"[ctx] cache: {N:,} Stichproben x {len(DTS)} dt; Region auf Saisontag geklemmt bei {outside:,}", flush=True)
    write_manifest(cfg, f"p34_ctx_features{'' if args.out_name == 'ctx_features' else '_' + args.out_name}", args.config, {
        "out_dir": str(out), "years": years, "px_names": PX_NAMES, "reg_names": REG_NAMES,
        "train_years": TRAIN_YEARS, "block_px": BLK, "nodata_rule": "FWI = 255 an >= 99 % der Trainings-Zelltage",
        "nodata_share_land": float(nodata[land].mean()), "lut_self_test_max_err": lut_err,
        "season_end": season_end, "sets": sets,
    })


if __name__ == "__main__":
    main()
