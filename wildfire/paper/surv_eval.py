"""Survival-Auswertung auf echten Stichtagen der Testjahre (nicht nach dem Paper vorsortiert).

Die PR-AUC auf den Paper-Stichproben bewertet einen Zeitpunkt auf einer stark
vorsortierten Stichprobe. Hier wird die ganze Risikokurve bewertet:

  Stichtage  zufaellige Landzelle in Kanada, zufaelliger Saisontag L
             (30-Tage-Fenster im Wuerfel, 1.4.-30.9.), Jahre --years;
             gleichverteilt, die Raten entsprechen also der Wirklichkeit
  Ziel       Zeit bis zum ersten MODIS-Feuer (MCD14A1-Rohraster, ganzes Jahr)
             nach L, in der Mittelzelle bzw. irgendwo im 13x13-Ausschnitt,
             administrativ zensiert bei 100 Tagen (L + 100 liegt immer in den
             Rohdaten, darum keine IPCW noetig, siehe wildfire.eval.metrics)
  Metriken   NLL, Integrated Brier, C-Index, ROC-AUC / PR-AUC / Brier / ECE
             je Horizont 1, 7, 14, 30, 60, 100 Tage (wildfire.eval.metrics)

Modelle (je Seed ein Checkpoint, Kopf als diskrete Hazards 1..100):
  km        Kaplan-Meier aus zufaelligen Stichtagen der Trainingsjahre
  km_fire   dasselbe getrennt nach Feuer im 30-Tage-Fenster (Ausschnitt bzw. Mitte)
  pxst      ST-DCTM pro Pixel (pixel_stdctm, Wuerfel-Vortraining); Ausschnitt
            ueber S_patch = prod S_pixel
  sts       STS-DCTM (sts_dctm); Mitte = Pixelkopf (6, 6), Ausschnitt = Ausschnittskopf

Die Stichtage werden einmal gebaut (--build) und unter derived_data/<out> abgelegt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from scipy.optimize import minimize

from wildfire.eval.metrics import evaluate, negative_log_likelihood
from wildfire.paper.pixel_landmarks import pixel_events
from wildfire.paper.pixel_stdctm import PixelSTDCTM
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import STSDCTM
from wildfire.paper.surv_prepare import DAY_BITS, HALF, T_WIN, window
from wildfire.paper.surv_train import km_hazard

H = 100
P = 2 * HALF + 1
DAY0 = dt.date(2000, 1, 1)
DAILY_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})\.tif{1,2}$", re.I)


def abs_day(d: dt.date) -> int:
    return (d - DAY0).days


def raw_fire_keys(root: Path):
    """Alle MODIS-Feuer-Pixeltage, ganzes Jahr: sortierte Schluessel pixel*2^15 + Tag seit 1.1.2000."""
    keys, last = [], 0
    for tar in sorted((root / "Entire_Canada_Maps" / "MCD14A1").glob("*.tar")):
        with tarfile.open(tar, "r:") as tf:
            for m in tf:
                hit = DAILY_RE.search(Path(m.name).name) if m.isfile() else None
                if not hit:
                    continue
                t = abs_day(dt.date(int(hit["y"]), int(hit["m"]), int(hit["d"])))
                last = max(last, t)
                with MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as ds:
                    a = ds.read(1)
                    rr, cc = np.nonzero(a == 1)
                    W = a.shape[1]
                if rr.size:
                    keys.append((rr.astype(np.int64) * W + cc) * DAY_BITS + t)
    k = np.sort(np.concatenate(keys))
    print(f"[seval] {k.size:,} MODIS-Feuer-Pixeltage, Rohdaten bis {DAY0 + dt.timedelta(days=last)}", flush=True)
    return k, last


def fire_in_window(keys, Ls, rr, cc, Hgrid, Wgrid, T=T_WIN):
    """Feuer irgendwo im Ausschnitt bzw. in der Mitte an den Tagen L-T+1 .. L."""
    dr, dc = np.meshgrid(np.arange(-HALF, HALF + 1), np.arange(-HALF, HALF + 1), indexing="ij")
    R, C = rr[:, None] + dr.ravel()[None], cc[:, None] + dc.ravel()[None]
    ok = (R >= 0) & (R < Hgrid) & (C >= 0) & (C < Wgrid)
    flat = np.where(ok, R * Wgrid + C, 0).astype(np.int64)
    pos = np.searchsorted(keys, flat * DAY_BITS + (Ls[:, None] - T + 1))
    nxt = np.where(pos < keys.size, keys[np.minimum(pos, keys.size - 1)], -1)
    hit = ok & (nxt // DAY_BITS == flat) & (nxt % DAY_BITS <= Ls[:, None])
    return hit.any(1), hit[:, (P * P) // 2]


def sample(cube, land, rng, years, n, last_raw):
    """Zufaellige Landzelle x Saisontag mit vollem Fenster im Wuerfel und L + H in den Rohdaten."""
    lr_, lc_ = np.nonzero(land)
    ok_L = []
    for L, d in enumerate(cube.dates):
        if int(d[:4]) not in years or L not in cube._pos or (L - T_WIN + 1) not in cube._pos:
            continue
        a, b = cube._pos[L - T_WIN + 1], cube._pos[L]
        if a[0] == b[0] and b[1] - a[1] == T_WIN - 1 and abs_day(dt.date.fromisoformat(d)) + H <= last_raw:
            ok_L.append(L)
    ok_L = np.array(ok_L)
    pick = rng.integers(0, lr_.size, size=n)
    return lr_[pick], lc_[pick], ok_L[rng.integers(0, ok_L.size, size=n)]


def build(args, cfg, exp, parts):
    cube = CubeReader(exp)
    keys, last_raw = raw_fire_keys(Path(cfg["dataset_root"]))
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    absL = lambda Lc: np.array([abs_day(dt.date.fromisoformat(cube.dates[int(x)])) for x in Lc])  # noqa: E731
    out = exp / "derived_data" / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    spec = {"eval": (args.years, args.n, True, 0), "km": (args.km_years, args.n_km, False, 1),
            "val": (args.val_years, args.n_val, True, 2)}
    for part in parts:
        years, n, with_x, off = spec[part]
        rng = np.random.default_rng(int(cfg["seed"]) + 31 + 1000 * off)
        yrs = set(range(years[0], years[1] + 1))
        rr, cc, Lc = sample(cube, land, rng, yrs, n, last_raw)
        La = absL(Lc)
        ev, t = pixel_events(keys, La, rr, cc, cube.H, cube.W)              # (n, 13, 13), bis 100 Tage
        pe, pt = ev.reshape(n, -1).any(1), t.reshape(n, -1).min(1)
        fw_patch, fw_centre = fire_in_window(keys, La, rr, cc, cube.H, cube.W)
        doy = np.array([dt.date.fromisoformat(cube.dates[int(x)]).timetuple().tm_yday for x in Lc], np.float32)
        np.savez(out / f"{part}_meta.npz", row=rr, col=cc, L_cube=Lc, L_abs=La, doy=doy,
                 ev_centre=ev[:, HALF, HALF], t_centre=t[:, HALF, HALF], ev_patch=pe, t_patch=pt,
                 fire_win_patch=fw_patch, fire_win_centre=fw_centre)
        print(f"[seval] {part} {years}: {n:,} Stichtage; Ereignis bis 100 Tage: Mitte {ev[:, HALF, HALF].mean():.4f}, "
              f"Ausschnitt {pe.mean():.4f}; Feuer im Fenster (Ausschnitt) {fw_patch.mean():.4f}", flush=True)
        if with_x:
            X = np.lib.format.open_memmap(out / f"{part}_X.npy", mode="w+", dtype=np.uint8, shape=(n, T_WIN, 55, P, P))
            for m, j in enumerate(np.argsort(Lc, kind="stable")):
                X[j] = window(cube, int(Lc[j]), int(rr[j]), int(cc[j]))
                if m % 20000 == 0:
                    print(f"    {m}/{n}", flush=True)
            X.flush(); del X
            # Selbsttest: Feuer im Fenster aus den Rohdaten = Feuerkanal des Wuerfelfensters
            Xm = np.load(out / f"{part}_X.npy", mmap_mode="r")
            chk = np.arange(0, n, max(1, n // 2000))
            fire_cube = np.asarray(Xm[chk, :, 0]).reshape(chk.size, -1).max(1) > 0
            agree = float((fire_cube == fw_patch[chk]).mean())
            print(f"[seval] Selbsttest Feuer im Fenster, Rohdaten gegen Wuerfel: {agree:.4f}", flush=True)
            if agree < 0.98:
                raise SystemExit("ABBRUCH: Wuerfel und Rohdaten passen nicht zusammen")


@torch.no_grad()
def hazards_pxst(ck, X, doy, mean, std, device, bs=128):
    c = torch.load(ck, map_location="cpu")
    net = PixelSTDCTM(degree=c["degree"], tau_max=c["tau_max"]).to(device)
    net.load_state_dict(c["state"]); net.eval()
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    tau = torch.arange(0, H + 1, device=device, dtype=torch.float64)
    hc, hp = [], []
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].float() - m_t) / s_t
        raw = net(x)                                            # (b, 13, 13, M+1)
        ls = net.dctm.log_survival(raw.reshape(-1, raw.shape[-1]), tau).view(raw.shape[0], P * P, H + 1)
        lsc = ls[:, (P * P) // 2]                               # Mitte
        lsp = ls.sum(1)                                         # Ausschnitt: S = prod S_pixel
        hc.append((-torch.expm1(lsc[:, 1:] - lsc[:, :-1])).cpu().numpy())
        hp.append((-torch.expm1(lsp[:, 1:] - lsp[:, :-1])).cpu().numpy())
    return np.concatenate(hc), np.concatenate(hp)


@torch.no_grad()
def hazards_sts(ck, X, doy, mean, std, device, bs=64):
    c = torch.load(ck, map_location="cpu")
    a_ = c.get("args", {})
    net = STSDCTM(d=a_.get("d", 96), heads=a_.get("heads", 4), depth=a_.get("depth", 4)).to(device)
    net.load_state_dict(c["state"]); net.eval()
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    hc, hp = [], []
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].float() - m_t) / s_t
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, raw_pa = net(x, torch.from_numpy(doy[a:a + bs]).to(device))
        hc.append(net.px_head.hazards(raw_px[:, HALF, HALF], H).double().cpu().numpy())
        hp.append(net.pa_head.hazards(raw_pa, H).double().cpu().numpy())
    return np.concatenate(hc), np.concatenate(hp)


def _cum_hazard(h: np.ndarray) -> np.ndarray:
    return -np.cumsum(np.log1p(-np.clip(h, 1e-12, 1 - 1e-7)), axis=1)


def calib_apply(h: np.ndarray, a: float, b: float) -> np.ndarray:
    """Lambda'(tau) = e^a * Lambda(tau)^b -> diskrete Hazards; monoton in tau, S(0) = 1."""
    lam = np.exp(a) * np.power(np.maximum(_cum_hazard(h), 1e-300), b)
    inc = np.diff(np.concatenate([np.zeros((lam.shape[0], 1)), lam], 1), axis=1)
    return np.clip(-np.expm1(-inc), 1e-15, 1 - 1e-7)


def calib_fit(h: np.ndarray, event: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """a, b per Survival-Likelihood; b = e^lb > 0 haelt die Rangfolge je Horizont."""
    f = lambda p: negative_log_likelihood(calib_apply(h, p[0], float(np.exp(p[1]))), event, t)  # noqa: E731
    r = minimize(f, x0=np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 400})
    return float(r.x[0]), float(np.exp(r.x[1]))


def ensemble(hs: list[np.ndarray]) -> np.ndarray:
    """Mittel der Survival-Kurven mehrerer Modelle -> diskrete Hazards."""
    S = np.mean([np.exp(-_cum_hazard(h)) for h in hs], axis=0)
    prev = np.concatenate([np.ones((S.shape[0], 1)), S[:, :-1]], 1)
    return np.clip(1 - S / np.maximum(prev, 1e-300), 1e-15, 1 - 1e-7)

def compact(m: dict) -> dict:
    out = {"nll": m["nll"], "ibs": m["integrated_brier"], "c_index": m["concordance_index"],
           "n": m["n"], "n_events": m["n_events"]}
    for k, v in m["horizons"].items():
        out.update({f"auc@{k}": v["roc_auc"], f"pr@{k}": v["pr_auc"], f"brier@{k}": v["brier"], f"ece@{k}": v["ece"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--years", nargs=2, type=int, default=[2023, 2025])
    ap.add_argument("--km-years", nargs=2, type=int, default=[2000, 2019])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--n-km", type=int, default=200_000)
    ap.add_argument("--val-years", nargs=2, type=int, default=[2020, 2022])
    ap.add_argument("--n-val", type=int, default=50_000)
    ap.add_argument("--calibrate", action="store_true",
                    help="zusaetzlich nachtraeglich kalibrieren: Lambda' = e^a Lambda^b, angepasst auf den Val-Stichtagen")
    ap.add_argument("--out-name", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G", help="Normierung wie im Training")
    ap.add_argument("--models", nargs="+", default=[
        "pxst:/data/Wildfire/checkpoints/paper/pxst_cache_G_cubeP_s*_s*.pt",
        "sts:/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_d128_s*_s*.pt"],
        help="art:glob je Modell; art = pxst | sts")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = exp / "derived_data" / args.out_name
    need = [p for p in ("eval", "km", "val") if args.build or not (out / f"{p}_meta.npz").exists()]
    if need:
        build(args, cfg, exp, need)

    ev = dict(np.load(out / "eval_meta.npz"))
    km = dict(np.load(out / "km_meta.npz"))
    targets_ = {"centre": (ev["ev_centre"], ev["t_centre"]), "patch": (ev["ev_patch"], ev["t_patch"])}
    res: dict = {}

    def record(model, seed_name, target, h):
        e, t = targets_[target]
        m = compact(evaluate(h, e.astype(bool), t.astype(int)))
        res.setdefault(model, {}).setdefault(target, {})[seed_name] = m
        print(f"[seval] {model:8s} {seed_name:10s} {target:6s} C {m['c_index']:.3f}  IBS {m['ibs']:.5f}  NLL {m['nll']:.4f}  "
              + "  ".join(f"AUC@{k} {m[f'auc@{k}']:.3f} PR@{k} {m[f'pr@{k}']:.3f}" for k in (7, 30, 100)), flush=True)

    # ---- Kaplan-Meier aus zufaelligen Stichtagen der Trainingsjahre
    for target, (ek, tk, fk, fe) in {"centre": ("ev_centre", "t_centre", "fire_win_centre", "fire_win_centre"),
                                      "patch": ("ev_patch", "t_patch", "fire_win_patch", "fire_win_patch")}.items():
        n = ev[ek].size
        record("km", "fit", target, np.tile(km_hazard(km[ek], km[tk]), (n, 1)))
        hs = {f: km_hazard(km[ek][km[fk] == f], km[tk][km[fk] == f]) for f in (False, True)}
        record("km_fire", "fit", target, np.stack([hs[bool(f)] for f in ev[fe]]))

    # ---- neuronale Modelle
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    Xg = to_device_once(np.load(out / "eval_X.npy"), device, "eval_X")
    if args.calibrate:
        va = dict(np.load(out / "val_meta.npz"))
        Xv = to_device_once(np.load(out / "val_X.npy"), device, "val_X")
        vt = {"centre": (va["ev_centre"].astype(bool), va["t_centre"].astype(int)),
              "patch": (va["ev_patch"].astype(bool), va["t_patch"].astype(int))}
    calib: dict = {}
    for spec in args.models:
        kind, pattern = spec.split(":", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[seval] {spec}: keine Checkpoints", flush=True)
            continue
        fn = hazards_pxst if kind == "pxst" else hazards_sts
        ens = {"test": {"centre": [], "patch": []}, "val": {"centre": [], "patch": []}}

        def with_cal(name, sname, tg, hv, ht):
            record(name, sname, tg, ht)
            if args.calibrate:
                a, b = calib_fit(hv, *vt[tg])
                calib.setdefault(f"{name}_cal", {}).setdefault(tg, {})[sname] = {"a": a, "b": b}
                record(f"{name}_cal", sname, tg, calib_apply(ht, a, b))

        for f in files:
            t0 = time.time()
            hc, hp = fn(f, Xg, ev["doy"], mean, std, device)
            hvc, hvp = fn(f, Xv, va["doy"], mean, std, device) if args.calibrate else (None, None)
            sname = Path(f).stem.split("_")[-1]
            for tg, hv, ht in (("centre", hvc, hc), ("patch", hvp, hp)):
                with_cal(kind, sname, tg, hv, ht)
                ens["test"][tg].append(ht)
                if args.calibrate:
                    ens["val"][tg].append(hv)
            print(f"    {Path(f).name} in {time.time() - t0:.0f} s", flush=True)
        if len(files) > 1:                                   # Ensemble: Mittel der Survival-Kurven
            for tg in ("centre", "patch"):
                with_cal(f"{kind}_ens", "ens", tg, ensemble(ens["val"][tg]) if args.calibrate else None,
                         ensemble(ens["test"][tg]))

    summary = {m: {tg: {k: {"mean": float(np.mean([s[k] for s in seeds.values()])),
                             "std": float(np.std([s[k] for s in seeds.values()]))}
                        for k in next(iter(seeds.values()))}
                   for tg, seeds in by_t.items()} for m, by_t in res.items()}
    write_manifest(cfg, f"p16_surv_eval{args.tag}", args.config, {
        "landmarks": {"years": args.years, "n": int(ev["row"].size), "km_years": args.km_years,
                      "km_n": int(km["row"].size), "sampling": "zufaellige Landzelle x Saisontag, gleichverteilt",
                      "event_rate_100d": {"centre": float(ev["ev_centre"].mean()), "patch": float(ev["ev_patch"].mean())},
                      "fire_in_window": {"centre": float(ev["fire_win_centre"].mean()),
                                         "patch": float(ev["fire_win_patch"].mean())}},
        "models": args.models, "results": res, "summary": summary, "calibration": calib,
        "calibration_rule": "Lambda'(tau) = exp(a) * Lambda(tau)^b, a und b per Survival-Likelihood auf Val-Stichtagen "
                            f"{args.val_years}; monoton in tau, S(0) = 1, Rangfolge je Horizont unveraendert",
    })


if __name__ == "__main__":
    main()
