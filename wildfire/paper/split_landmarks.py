"""Wuerfel-Stichtage fuer Training und Survival-Auswertung auf einem Kandidaten-Split aus split_audit.

Die Ablation (sts_extreme / sts_focus / focus_eval) trainiert auf Wuerfel-Stichtagen 2000-2019 (paper_pixel) und wertet
die Survival-Metriken auf zufaelligen Stichtagen Val 2020-2022 / Test 2023-2025 aus (surv_eval_test). Fuer year_strat
und block_strat entstehen hier dieselben Dateien, nur aus den erlaubten Raum-Zeit-Bereichen des Splits:

  year_strat   Stichtag L erlaubt fuer Teil s, wenn das Jahr von L zu s gehoert (Jahre aus derived_data/splits_v2).
  block_strat  Einheiten Jahr x --unit-px-Block wie in split_audit; Einheiten ohne Cache-Stichprobe zaehlen als train.
               train: Einheit train und mindestens --gap Pixel (Ausschnitt-Ueberlappung) von jeder val/test-Einheit
               desselben Jahres entfernt; val: Einheit val und --gap von test entfernt; test: Einheit test.
               Ein Stichtagsfenster [L - 29, L + 100] reicht von Maerz bis hoechstens Januar, Jahresgrenzen beruehren
               also keine Saison des Nachbarjahrs.
Zusaetzlich fuer alle Teile: L + 100 Tage liegt in den Rohdaten (sonst waere das Ziel faelschlich zensiert).

Erzeugung wie pixel_landmarks (Stroeme fire und random, gleiche Ziele, gleiche Fenster) bzw. wie surv_eval.build
(zufaellige Landzelle x Saisontag, Ziele aus den Rohdaten); es werden die Funktionen von dort benutzt.
Ausgabe NEU: derived_data/<out-cube>/ train_X.npy, train_event.npy, train_t.npy, train_meta.npz;
             derived_data/<out-surv>/ val_*, eval_* (meta.npz, X.npy); Manifest p50_split_landmarks_<name><tag>.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from scipy import ndimage

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.pixel_landmarks import H_FOLLOW, P, pixel_events
from wildfire.paper.split_audit import load_json
from wildfire.paper.surv_eval import abs_day, fire_in_window, raw_fire_keys, sample
from wildfire.paper.surv_prepare import DAY_BITS, HALF, T_WIN, fire_keys, window

PARTS = {"train": 0, "val": 1, "test": 2}


class Allowed:
    """allowed(part, L, r, c) -> bool-Array fuer Stichtage (Wuerfelindex L, Zelle r, c)."""

    def __init__(self, name: str, split: np.ndarray, base: dict, years_of_L: np.ndarray, H: int, W: int,
                 unit: int, gap: int):
        self.name, self.year_of = name, years_of_L
        keep = split != "drop"
        if name == "year_strat":
            self.part_of_year = {}
            for y, s in zip(base["year"][keep], split[keep]):
                if self.part_of_year.setdefault(int(y), s) != s:
                    raise SystemExit(f"ABBRUCH: Jahr {y} liegt in mehreren Splits - kein Jahres-Split")
            return
        nbr, nbc = -(-H // unit), -(-W // unit)
        lab = {}
        for y, r, c, s in zip(base["year"][keep], base["row"][keep], base["col"][keep], split[keep]):
            g = lab.setdefault(int(y), np.zeros((nbr, nbc), np.int8))
            g[r // unit, c // unit] = max(g[r // unit, c // unit], PARTS[s])
        st = np.ones((2 * gap + 1, 2 * gap + 1), bool)
        self.px = {}
        for y, g in lab.items():
            full = np.kron(g, np.ones((unit, unit), np.int8))[:H, :W]
            near_test = ndimage.binary_dilation(full == 2, st)
            near_val = ndimage.binary_dilation(full == 1, st)
            m = np.zeros((H, W), np.int8) - 1                               # -1 = gesperrt
            m[(full == 0) & ~near_val & ~near_test] = 0
            m[(full == 1) & ~near_test] = 1
            m[full == 2] = 2
            self.px[y] = m
        self.H, self.W = H, W

    def __call__(self, part: str, L: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        yr = self.year_of[np.asarray(L, np.int64)]
        if self.name == "year_strat":
            return np.array([self.part_of_year.get(int(y)) == part for y in yr], bool)
        out = np.zeros(len(yr), bool) if part != "train" else np.ones(len(yr), bool)
        for y in np.unique(yr):
            i = np.nonzero(yr == y)[0]
            m = self.px.get(int(y))
            if m is None:                                                   # Jahr ohne Einheiten: alles train
                out[i] = part == "train"
                continue
            out[i] = m[r[i], c[i]] == PARTS[part]
        return out


def build_train(args, cfg, exp, cube, ok, part_years, last_day, rng):
    H, W = cube.H, cube.W
    t_of = {d: i for i, d in enumerate(cube.dates)}
    year_of = np.array([int(d[:4]) for d in cube.dates])
    keys = fire_keys(Path(cfg["dataset_root"]), t_of, W)

    def landmark_ok(L: int) -> bool:
        return (L in cube._pos and (L - T_WIN + 1) in cube._pos
                and cube._pos[L - T_WIN + 1][0] == cube._pos[L][0]
                and cube._pos[L - T_WIN + 1][1] == cube._pos[L][1] - T_WIN + 1
                and L + H_FOLLOW <= last_day)

    valid_L = np.array(sorted(L for L in cube._pos if year_of[L] in part_years and landmark_ok(L)))
    valid_set = set(valid_L.tolist())
    print(f"[slm] train: {valid_L.size} gueltige Stichtage {cube.dates[valid_L[0]]} .. {cube.dates[valid_L[-1]]}",
          flush=True)

    k_day, k_pix = keys % DAY_BITS, keys // DAY_BITS
    tr_fire = np.where(np.isin(year_of[np.minimum(k_day, len(year_of) - 1)], list(part_years)))[0]
    fr, fc, fL, fu, far, fac = [], [], [], [], [], []
    while len(fr) < args.n_fire:
        j = rng.choice(tr_fire, size=args.n_fire)
        u = rng.integers(1, H_FOLLOW + 1, size=j.size)
        L = k_day[j] - u
        a_r, a_c = k_pix[j] // W, k_pix[j] % W
        r = a_r + rng.integers(-HALF, HALF + 1, size=j.size)
        c = a_c + rng.integers(-HALF, HALF + 1, size=j.size)
        keep = np.array([int(x) in valid_set for x in L]) & (r >= 0) & (r < H) & (c >= 0) & (c < W)
        keep[keep] = ok("train", L[keep], r[keep], c[keep])
        fr += r[keep].tolist(); fc += c[keep].tolist(); fL += L[keep].tolist(); fu += u[keep].tolist()
        far += a_r[keep].tolist(); fac += a_c[keep].tolist()
    fr, fc, fL, fu, far, fac = (np.array(a[:args.n_fire]) for a in (fr, fc, fL, fu, far, fac))

    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    lr_, lc_ = np.nonzero(land)
    rr_, rc_, rL = [], [], []
    while len(rr_) < args.n_random:
        pick = rng.integers(0, lr_.size, size=args.n_random)
        L = valid_L[rng.integers(0, valid_L.size, size=args.n_random)]
        keep = ok("train", L, lr_[pick], lc_[pick])
        rr_ += lr_[pick][keep].tolist(); rc_ += lc_[pick][keep].tolist(); rL += L[keep].tolist()
    rr_, rc_, rL = (np.array(a[:args.n_random]) for a in (rr_, rc_, rL))

    rows = np.concatenate([fr, rr_]).astype(np.int64)
    cols = np.concatenate([fc, rc_]).astype(np.int64)
    Ls = np.concatenate([fL, rL]).astype(np.int64)
    kind = np.concatenate([np.zeros(fr.size, np.int8), np.ones(rr_.size, np.int8)])
    anchor_u = np.concatenate([fu, np.zeros(rr_.size, np.int64)])
    event, t = pixel_events(keys, Ls, rows, cols, H, W)
    pr, pc = far - fr + HALF, fac - fc + HALF
    i_f = np.arange(fr.size)
    anchor_ok = event[i_f, pr, pc] & (t[i_f, pr, pc] <= fu)
    print(f"[slm] Ankerpixel brennt spaetestens nach u Tagen: {anchor_ok.mean():.4f}", flush=True)
    if anchor_ok.mean() < 0.999:
        raise SystemExit("ABBRUCH: Ereignisse passen nicht zu den Ankern - Zeitindex pruefen")

    out = exp / "derived_data" / args.out_cube
    out.mkdir(parents=True)
    n = rows.size
    X = np.lib.format.open_memmap(out / "train_X.npy", mode="w+", dtype=np.uint8, shape=(n, T_WIN, 55, P, P))
    for m, j in enumerate(np.argsort(Ls, kind="mergesort")):
        X[j] = window(cube, int(Ls[j]), int(rows[j]), int(cols[j]))
        if m % 20000 == 0:
            print(f"    train {m}/{n}", flush=True)
    X.flush(); del X
    np.save(out / "train_event.npy", event)
    np.save(out / "train_t.npy", t)
    np.savez(out / "train_meta.npz", row=rows, col=cols, L=Ls, kind=kind, anchor_u=anchor_u)
    yrs = np.bincount(year_of[Ls] - 2000, minlength=26)
    return {"n": int(n), "n_fire_stream": int(fr.size), "n_random_stream": int(rr_.size),
            "bytes": human_bytes(n * T_WIN * 55 * P * P),
            "years": {int(2000 + i): int(v) for i, v in enumerate(yrs) if v},
            "pixel_event_share": {"fire": float(event[kind == 0].mean()), "random": float(event[kind == 1].mean())},
            "landmarks_with_any_event": {"fire": float(event[kind == 0].any((1, 2)).mean()),
                                         "random": float(event[kind == 1].any((1, 2)).mean())}}


def build_surv(args, cfg, exp, cube, ok, years_by_part, part, fname, n, seed_off, keys, last_raw):
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    rng = np.random.default_rng(int(cfg["seed"]) + 31 + 1000 * seed_off)
    rr, cc, Lc = [], [], []
    while len(rr) < n:
        r, c, L = sample(cube, land, rng, years_by_part[part], n, last_raw)
        keep = ok(part, L, r, c)
        rr += r[keep].tolist(); cc += c[keep].tolist(); Lc += L[keep].tolist()
        print(f"[slm] {fname}: {len(rr):,} / {n:,} Stichtage", flush=True)
    rr, cc, Lc = (np.array(a[:n], np.int64) for a in (rr, cc, Lc))
    La = np.array([abs_day(dt.date.fromisoformat(cube.dates[int(x)])) for x in Lc])
    ev, t = pixel_events(keys, La, rr, cc, cube.H, cube.W)
    pe, pt = ev.reshape(n, -1).any(1), t.reshape(n, -1).min(1)
    fw_patch, fw_centre = fire_in_window(keys, La, rr, cc, cube.H, cube.W)
    doy = np.array([dt.date.fromisoformat(cube.dates[int(x)]).timetuple().tm_yday for x in Lc], np.float32)
    out = exp / "derived_data" / args.out_surv
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{fname}_meta.npz", row=rr, col=cc, L_cube=Lc, L_abs=La, doy=doy,
             ev_centre=ev[:, HALF, HALF], t_centre=t[:, HALF, HALF], ev_patch=pe, t_patch=pt,
             fire_win_patch=fw_patch, fire_win_centre=fw_centre)
    X = np.lib.format.open_memmap(out / f"{fname}_X.npy", mode="w+", dtype=np.uint8, shape=(n, T_WIN, 55, P, P))
    for m, j in enumerate(np.argsort(Lc, kind="stable")):
        X[j] = window(cube, int(Lc[j]), int(rr[j]), int(cc[j]))
        if m % 20000 == 0:
            print(f"    {fname} {m}/{n}", flush=True)
    X.flush(); del X
    Xm = np.load(out / f"{fname}_X.npy", mmap_mode="r")
    chk = np.arange(0, n, max(1, n // 2000))
    agree = float(((np.asarray(Xm[chk, :, 0]).reshape(chk.size, -1).max(1) > 0) == fw_patch[chk]).mean())
    print(f"[slm] {fname}: Ereignis bis 100 d Mitte {ev[:, HALF, HALF].mean():.4f}, Ausschnitt {pe.mean():.4f}; "
          f"Selbsttest Feuer im Fenster {agree:.4f}", flush=True)
    if agree < 0.98:
        raise SystemExit("ABBRUCH: Wuerfel und Rohdaten passen nicht zusammen")
    yrs = np.bincount(np.array([int(cube.dates[int(x)][:4]) for x in Lc]) - 2000, minlength=26)
    return {"n": int(n), "event_patch": float(pe.mean()), "event_centre": float(ev[:, HALF, HALF].mean()),
            "selftest": agree, "years": {int(2000 + i): int(v) for i, v in enumerate(yrs) if v}}


def leak_check(exp, args, cube, base, split):
    """Abstand der Trainings-Stichtage zu Val/Test (Cache-Stichproben und Survival-Stichtage) - muss leer sein."""
    from wildfire.paper.split_audit import near
    m = np.load(exp / "derived_data" / args.out_cube / "train_meta.npz")
    day = lambda L: np.array([abs_day(dt.date.fromisoformat(cube.dates[int(x)])) for x in L])  # noqa: E731
    tr = np.c_[m["row"], m["col"], day(m["L"]) + 35].astype(float)      # Fenstermitte von [L - 29, L + 100]
    res = {}
    for s in ("val", "test"):
        c = split == s
        P_c = np.c_[base["row"][c], base["col"][c], base["day"][c] - 15].astype(float)   # Mitte von [D - 60, D + 30]
        sm = np.load(exp / "derived_data" / args.out_surv / f"{'eval' if s == 'test' else 'val'}_meta.npz")
        P_s = np.c_[sm["row"], sm["col"], sm["L_abs"] + 35].astype(float)
        # Ueberlappung, wenn |dr|, |dc| <= 12 und die Zeitfenster sich schneiden (Halbbreiten 65 + 46 bzw. 65 + 65)
        res[s] = {"cache": float(near(tr, P_c, 2 * HALF, 65 + 46).mean()),
                  "surv": float(near(tr, P_s, 2 * HALF, 130).mean())}
    print(f"[slm] Leck-Check (Anteil Trainings-Stichtage nahe Val/Test): {res}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", choices=["year_strat", "block_strat"], required=True)
    ap.add_argument("--split-tag", default="")
    ap.add_argument("--unit", type=int, default=64)
    ap.add_argument("--gap", type=int, default=2 * HALF)
    ap.add_argument("--n-fire", type=int, default=120_000)
    ap.add_argument("--n-random", type=int, default=120_000)
    ap.add_argument("--n-val", type=int, default=50_000)
    ap.add_argument("--n-eval", type=int, default=100_000)
    ap.add_argument("--out-cube", required=True)
    ap.add_argument("--out-surv", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    for d in (args.out_cube, args.out_surv):
        if (exp / "derived_data" / d).exists():
            raise SystemExit(f"ABBRUCH: derived_data/{d} existiert schon")

    base = load_json(Path(cfg["dataset_root"]) / "cache_G" / "samples_variant_G.json")
    split = np.load(exp / "derived_data" / f"splits_v2{args.split_tag}" / args.name / "split.npy")
    if len(split) != len(base["y"]):
        raise SystemExit("ABBRUCH: split.npy passt nicht zu cache_G")
    cube = CubeReader(exp)
    year_of = np.array([int(d[:4]) for d in cube.dates])
    ok = Allowed(args.name, split, base, year_of, cube.H, cube.W, args.unit, args.gap)
    allyears = set(range(2000, 2026))
    years_by_part = ({p: {y for y, s in ok.part_of_year.items() if s == p} for p in PARTS}
                     if args.name == "year_strat" else {p: allyears for p in PARTS})
    print(f"[slm] {args.name}: Jahre je Teil {({p: sorted(v) for p, v in years_by_part.items()})}", flush=True)
    raw_keys, last_raw = raw_fire_keys(Path(cfg["dataset_root"]))
    last_day = max(i for i, d in enumerate(cube.dates) if abs_day(dt.date.fromisoformat(d)) <= last_raw)

    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    info = {"train": build_train(args, cfg, exp, cube, ok, years_by_part["train"], last_day, rng)}
    info["val"] = build_surv(args, cfg, exp, cube, ok, years_by_part, "val", "val", args.n_val, 2, raw_keys, last_raw)
    info["test"] = build_surv(args, cfg, exp, cube, ok, years_by_part, "test", "eval", args.n_eval, 0, raw_keys,
                              last_raw)
    info["leak"] = leak_check(exp, args, cube, base, split)
    write_manifest(cfg, f"p50_split_landmarks_{args.name}{args.tag}", args.config, {
        "args": vars(args), "info": info,
        "rule": "year_strat: Jahr von L; block_strat: Einheit Jahr x unit-Block, train/val mit gap Abstand zu val/test",
        "outputs": {"cube": str(exp / "derived_data" / args.out_cube), "surv": str(exp / "derived_data" / args.out_surv)}})


if __name__ == "__main__":
    main()
