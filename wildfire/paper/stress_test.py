"""Stresstest "Was waere wenn": wie reagiert das Modell auf kuenstlich verschaerftes Feuerwetter?

Anlass ist die Kritik, datengetriebene Modelle saehen beispiellose Lagen nicht (paper_notes/
reviewer_extremes_wui_2026-10-24.md). Kein Neutraining: die fertigen Modelle des Ablationsprotokolls
(STS-LH und Contagion-STS ohne Warmstart, je 3 Seeds) bekommen die echten Test-Stichtage aus
surv_eval_test (2023-2025), aber mit veraenderten Feuerwetter-Kanaelen. Veraendert wird in den
Rohwerten (uint8-Einheiten des Wuerfels) VOR der Normierung und OHNE Kappung bei 255, so dass auch
Werte jenseits des darstellbaren Bereichs (Extrapolation) eingehen. Alle anderen Kanaele bleiben gleich.

  shift   Gruppe um Delta verschoben, ganzes 30-Tage-Fenster, alle Pixel des Ausschnitts (unten bei 0
          begrenzt): Duerre DMC/DC/BUI, Feuerwetter FFMC/ISI/FWI, Hitze t2m_max/VPD, alle acht zusammen
  level   ein Kanal auf einen festen Wert gesetzt (Trainings-p50/p90/p99 der Stichtagszellen, 255, 280,
          320), nur am Stichtag (last) oder im ganzen Fenster (window)
  code    FWI und/oder ISI im ganzen Fenster auf 255 (NoData- bzw. Saettigungs-Code) gegen 254

Pixel, deren Kanal im ganzen Fenster 255 ist (NoData bei FWI/ISI, siehe fwi255_check), bleiben bei
shift und level unveraendert. Stichtage: feste Zufallsauswahl (natuerliche Haeufigkeit) plus alle
Stichtage mit DC >= Trainings-p99 oder DC ueber dem Maximum derselben Zelle 2000-2019 (Klimatologie
aus extreme_eval, derived_data/extreme_eval/).

Zwei Schritte:
  --ckpt     je Checkpoint (GPU, Array-Job): F(7), F(30), F(100) fuer Mitte (Pixelkopf 6, 6) und
             Ausschnitt je Szenario und Stichtag -> derived_data/stress_test/<modell>_<seed><tag>.npz
  --summarize  (CPU) Ensemble (Mittel der Survival-Kurven = Mittel von F) und Seeds, Kennzahlen je
             Gruppe -> Manifest p32_stress_test<tag>

Kopf-Bias wie surv_eval (load_sts, Modus default): absolute F sind darum nur bedingt deutbar, berichtet
werden vor allem relative Aenderungen und die Richtung (Monotonie).
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.extreme_eval import WEATHER, train_rank
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import load_sts
from wildfire.paper.surv_prepare import HALF

TAUS = (7, 30, 100)
DELTAS = (-40, -20, 20, 40, 60, 80, 120)
SHIFT_GROUPS = {"drought": ("DMC", "DC", "BUI"), "fireweather": ("FFMC", "ISI", "FWI"),
                "heat": ("t2m_max", "VPD"), "all": tuple(WEATHER)}
LEVELS = ("p50", "p90", "p99", 255, 280, 320)
TARGETS = ("patch", "centre")


def scenarios(thr: dict) -> list[dict]:
    """Liste der Szenarien; Index 0 ist der unveraenderte Stichtag."""
    out = [{"name": "base", "kind": "base"}]
    for g, chs in SHIFT_GROUPS.items():
        for d in DELTAS:
            out.append({"name": f"shift_{g}_{d:+d}", "kind": "shift", "group": g,
                        "ch": [WEATHER[c] for c in chs], "delta": d})
    for c in WEATHER:
        for mode in ("last", "window"):
            for lv in LEVELS:
                v = thr[c][lv] if isinstance(lv, str) else lv
                out.append({"name": f"level_{c}_{mode}_{lv}", "kind": "level", "chname": c, "ch": [WEATHER[c]],
                            "mode": mode, "level": str(lv), "value": float(v)})
    for chs in (("FWI",), ("ISI",), ("FWI", "ISI")):
        for v in (254, 255):
            out.append({"name": f"code_{'_'.join(chs)}_{v}", "kind": "code", "ch": [WEATHER[c] for c in chs],
                        "value": float(v)})
    return out


def apply(x: torch.Tensor, sc: dict) -> torch.Tensor:
    """x (B, 30, 55, 13, 13) Rohwerte float -> veraenderte Kopie."""
    if sc["kind"] == "base":
        return x
    x = x.clone()
    for ch in sc["ch"]:
        v = x[:, :, ch]                                         # (B, 30, 13, 13)
        if sc["kind"] == "code":
            v.fill_(sc["value"])
            continue
        nodata = (v == 255).all(1, keepdim=True)                # Pixel ohne Daten im ganzen Fenster
        if sc["kind"] == "shift":
            new = (v + sc["delta"]).clamp(min=0)
        elif sc["mode"] == "window":
            new = torch.full_like(v, sc["value"])
        else:
            new = v.clone()
            new[:, -1] = sc["value"]
        x[:, :, ch] = torch.where(nodata, v, new)
    return x


def select(cfg: dict, exp: Path, n_random: int, limit: int):
    """Stichtage: Zufallsauswahl + Extremgruppen; dazu die Schwellen der Level-Szenarien."""
    d = exp / "derived_data" / "surv_eval_test"
    te = dict(np.load(d / "eval_meta.npz"))
    cl = dict(np.load(exp / "derived_data" / "extreme_eval" / "cell_clim_surv_eval_test_step1.npz"))
    ranks = train_rank(cl)
    names = list(WEATHER)
    thr = {c: {f"p{int(q * 100)}": int(np.searchsorted(ranks[j], q)) for q in (0.5, 0.9, 0.99)}
           for j, c in enumerate(names)}
    X = np.load(d / "eval_X.npy", mmap_mode="r")
    jd = names.index("DC")
    dc = np.asarray(X[:, -1, WEATHER["DC"], HALF, HALF])       # Stichtag, Mitte
    W = CubeReader(exp).W
    ci = np.searchsorted(cl["cells"], te["row"] * W + te["col"])
    if not np.array_equal(cl["cells"][ci], te["row"] * W + te["col"]):
        raise SystemExit("ABBRUCH: Stichtagszellen fehlen in der Klimatologie")
    ge_p99 = ranks[jd][dc] >= 0.99
    above = dc > cl["vmax"][ci, jd]
    rnd = np.zeros(dc.size, bool)
    rnd[np.random.default_rng(2026).choice(dc.size, min(n_random, dc.size), replace=False)] = True
    rows = np.where(rnd | ge_p99 | above)[0]
    if limit:
        rows = rows[np.random.default_rng(1).choice(rows.size, min(limit, rows.size), replace=False)]
        rows.sort()
    flags = {"random": rnd[rows], "dc_ge_p99": ge_p99[rows], "dc_above_cell_max": above[rows]}
    return rows, flags, thr, te, X


@torch.no_grad()
def run_ckpt(args, cfg, exp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, flags, thr, te, Xm = select(cfg, exp, args.n_random, args.limit)
    scs = scenarios(thr)
    t0 = time.time()
    X = torch.from_numpy(np.asarray(Xm[rows])).to(device)     # uint8, ausgewaehlte Stichtage
    doy = torch.from_numpy(te["doy"][rows].astype(np.float32)).to(device)
    print(f"[stress] {rows.size:,} Stichtage ({flags['random'].sum():,} zufaellig, DC >= p99 "
          f"{flags['dc_ge_p99'].sum():,}, ueber Zellmax {flags['dc_above_cell_max'].sum():,}), "
          f"{len(scs)} Szenarien, geladen in {time.time() - t0:.0f} s", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / "cache_G" / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    net = load_sts(args.ckpt, device)
    tau = torch.tensor(TAUS, device=device, dtype=torch.float64)
    out = {tg: np.zeros((len(scs), rows.size, len(TAUS)), np.float32) for tg in TARGETS}
    for si, sc in enumerate(scs):
        t1 = time.time()
        for a in range(0, rows.size, args.bs):
            x = apply(X[a:a + args.bs].float(), sc)
            raw_px, raw_pa = net((x - m_t) / s_t, doy[a:a + args.bs])     # float32, kein autocast
            fc = -torch.expm1(net.px_head.log_survival(raw_px[:, HALF, HALF], tau))
            fp = -torch.expm1(net.pa_head.log_survival(raw_pa, tau))
            out["centre"][si, a:a + args.bs] = fc.float().cpu().numpy()
            out["patch"][si, a:a + args.bs] = fp.float().cpu().numpy()
        if si < 3 or si % 20 == 0:
            print(f"    {si:3d} {sc['name']:32s} F30 Ausschnitt {out['patch'][si, :, 1].mean():.4f} "
                  f"Mitte {out['centre'][si, :, 1].mean():.5f}  ({time.time() - t1:.0f} s)", flush=True)
    od = exp / "derived_data" / "stress_test"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{args.model_name}_{Path(args.ckpt).stem.split('_')[-1]}{args.tag}.npz"
    np.savez(p, rows=rows, names=np.array([s["name"] for s in scs]), F_patch=out["patch"], F_centre=out["centre"],
             **{f"flag_{k}": v for k, v in flags.items()})
    print(f"[stress] {p} ({time.time() - t0:.0f} s)", flush=True)


def lam(F: np.ndarray) -> np.ndarray:
    return -np.log1p(-np.clip(F, 0, 1 - 1e-7))


def chain_stats(Fs: list[np.ndarray], m: np.ndarray, tol: float) -> dict:
    """Folge F30 bei steigender Eingabe: Anteil Stichtage mit Abfall zwischen Nachbarstufen (streng und um
    mehr als tol relativ) und mit mindestens einem Abfall irgendwo in der Kette."""
    S = np.stack([F[m] for F in Fs])                            # (Stufen, n)
    dec = S[1:] < S[:-1]
    dec_tol = S[1:] < S[:-1] * (1 - tol)
    return {"steps_strict": [float(v) for v in dec.mean(1)], "steps_tol": [float(v) for v in dec_tol.mean(1)],
            "any_strict": float(dec.any(0).mean()), "any_tol": float(dec_tol.any(0).mean())}


def summarize(args, exp, level_value: dict):
    od = exp / "derived_data" / "stress_test"
    res, files = {}, {}
    for name in args.models:
        fs = sorted(glob.glob(str(od / f"{name}_s[0-9]{args.tag}.npz")))
        if not fs:
            print(f"[stress] {name}: keine Dateien", flush=True)
            continue
        zs = [dict(np.load(f)) for f in fs]
        names = [str(s) for s in zs[0]["names"]]
        flags = {k[5:]: zs[0][k] for k in zs[0] if k.startswith("flag_")}
        grp = {"alle_zufaellig": flags["random"], "DC_ge_p99": flags["dc_ge_p99"],
               "DC_ueber_zell_max": flags["dc_above_cell_max"]}
        files[name] = [Path(f).name for f in fs]
        variants = {Path(f).stem[len(name) + 1:len(name) + 3]: z for f, z in zip(fs, zs)}
        if len(zs) > 1:
            variants["ens"] = {tg: np.mean([z[tg] for z in zs], 0) for tg in ("F_patch", "F_centre")}
        res[name] = {}
        for vn, z in variants.items():
            r = res[name][vn] = {}
            for tg in TARGETS:
                Fa = z[f"F_{tg}"].astype(np.float64)              # (Szenarien, n, 3)
                rt = r[tg] = {}
                for gname, m in grp.items():
                    if m.sum() == 0:
                        continue
                    base = Fa[0][m]
                    g = rt[gname] = {"n": int(m.sum()), "scen": {}}
                    for si, sn in enumerate(names):
                        F = Fa[si][m]
                        dl = np.log(np.maximum(lam(F[:, 1]), 1e-12)) - np.log(np.maximum(lam(base[:, 1]), 1e-12))
                        g["scen"][sn] = {"F7": float(F[:, 0].mean()), "F30": float(F[:, 1].mean()),
                                         "F100": float(F[:, 2].mean()),
                                         "ratio_F30": float(F[:, 1].mean() / max(base[:, 1].mean(), 1e-12)),
                                         "dlogLam30_mean": float(dl.mean()), "dlogLam30_median": float(np.median(dl)),
                                         "share_F30_below_base": float((F[:, 1] < base[:, 1]).mean())}
                    ix = {sn: i for i, sn in enumerate(names)}
                    F30 = lambda sn: Fa[ix[sn]][:, 1]  # noqa: E731
                    g["chains"] = {}
                    for sg in SHIFT_GROUPS:              # Delta aufsteigend, 0 = base
                        seq = [f"shift_{sg}_{d:+d}" for d in DELTAS if d < 0] + ["base"] + \
                              [f"shift_{sg}_{d:+d}" for d in DELTAS if d > 0]
                        g["chains"][f"shift_{sg}"] = {"order": seq, **chain_stats([F30(s) for s in seq], m, args.tol)}
                    for c in WEATHER:
                        for mode in ("last", "window"):
                            lv = [s for s in names if s.startswith(f"level_{c}_{mode}_")]
                            seq = sorted(lv, key=lambda s: level_value[s])
                            g["chains"][f"level_{c}_{mode}"] = {
                                "order": seq, "values": [level_value[s] for s in seq],
                                **chain_stats([F30(s) for s in seq], m, args.tol)}
                    for chs in ("FWI", "ISI", "FWI_ISI"):
                        a, b = F30(f"code_{chs}_254")[m], F30(f"code_{chs}_255")[m]
                        g["chains"][f"code_{chs}_254_to_255"] = {
                            "ratio_F30": float(b.mean() / max(a.mean(), 1e-12)),
                            "share_255_below_254": float((b < a).mean()),
                            "share_255_below_254_tol": float((b < a * (1 - args.tol)).mean())}
                print(f"[stress] {name} {vn} {tg}: fertig", flush=True)
    return res, files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="", help="ein Checkpoint (Array-Job); leer mit --summarize")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", nargs="+", default=["lh", "cg_scratch"])
    ap.add_argument("--n-random", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="Rauchtest: so viele Stichtage")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--tol", type=float, default=0.01, help="relative Toleranz fuer einen Abfall von F30")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if not args.summarize:
        run_ckpt(args, cfg, exp)
        return
    _, _, thr, _, _ = select(cfg, exp, args.n_random, args.limit)
    scs = scenarios(thr)
    res, files = summarize(args, exp, {s["name"]: s["value"] for s in scs if s["kind"] == "level"})
    write_manifest(cfg, f"p32_stress_test{args.tag}", args.config, {
        "models": args.models, "files": files, "n_random": args.n_random, "limit": args.limit,
        "thresholds_uint8": thr, "deltas": DELTAS, "shift_groups": SHIFT_GROUPS, "levels": [str(v) for v in LEVELS],
        "taus": TAUS, "tol": args.tol, "scenarios": scs, "results": res,
        "notes": {
            "einheiten": "Rohwerte uint8 des Wuerfels, Aenderung vor der Normierung, keine Kappung bei 255",
            "nodata": "Pixel mit Kanal = 255 im ganzen Fenster bleiben bei shift/level unveraendert",
            "bias": "load_sts, Kopf-Bias default (wie surv_eval); absolute F nur bedingt deutbar",
            "ensemble": "Mittel der Survival-Kurven der Seeds = Mittel von F",
            "chains": "Anteil Stichtage mit Abfall von F30 zwischen benachbarten Stufen steigender Eingabe; "
                      "tol: Abfall um mehr als tol relativ",
            "gruppen": "alle_zufaellig = feste Zufallsauswahl (natuerliche Haeufigkeit); DC_ge_p99 / "
                       "DC_ueber_zell_max = alle Test-Stichtage dieser Gruppen (Wert am Stichtag, Mitte)",
        },
    })


if __name__ == "__main__":
    main()
