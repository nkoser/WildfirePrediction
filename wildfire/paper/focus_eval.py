"""Paper-Protokoll je dt auf allen Test-Stichproben und auf neuen Zuendungen (ign30), GPU.

cache_G_L61, Test: Stichtag e = D - dt, Score F(dt + 1), Ziel Feuer am Tag D + 1 (Mitte = Pixel (6, 6), Ausschnitt),
wie sts_extreme.paper_eval. Teilmengen je dt:
  all     alle Test-Stichproben
  ign30   kein Feuer im Ausschnitt an den 30 Fenstertagen e - 29 .. e (neue Zuendungen)
  fire30  der Rest (Feuer im Fenster)
Modelle (--models), je Seed aus checkpoints/paper/sts_extreme_<name>_s<seed>_s<seed>.pt, Kopf-Bias aus dem Checkpoint:
  name                     ohne Huelle (lh_ref, cg_ref, ...)
  name:inputs=<satz>       sts_inputs-Maske (fireonly, nofire)
  name:focus=<varianten>[:pad=reflect]   sts_focus-Huelle (fdrop/ignw wirken nur im Training; resid laedt
                           derived_data/sts_focus/coef_<name>_s<seed>.npz)
Dazu ohne GPU: logistische Feuerhistorie-Basis (sts_focus.features_px am Mittelpixel, je Ziel und tau = dt + 1
angepasst auf den Wuerfel-Stichtagen) und der U-Net-Nachbau (predictions/paper/rec_cache_G_cache_next_paperhp_w0.npz,
nur Ausschnitt). Kennzahl PR-AUC: Mittel ueber die Seeds und Ensemble (Mittel der Scores).
Ausgabe NEU: metrics/focus_eval<tag>/ (summary.md, results.json, scores.npz), Manifest p45_focus_eval<tag>.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper import sts_extreme, targets
from wildfire.paper.compare import DTS
from wildfire.paper.fire_baseline import fire_of
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import CENTRE, doy_of, patch_events
from wildfire.paper.sts_focus import _ORIG_BUILD, build_focus, features_px, load_coef

UNET_ALL = {1: 0.691, 7: 0.607, 15: 0.558, 30: 0.515}


def parse(spec: str) -> tuple[str, dict]:
    name, *opts = spec.split(":")
    return name, dict(o.split("=", 1) for o in opts)


def load_nets(exp: Path, spec: str, seeds, device, mu_f, sd_f):
    name, opt = parse(spec)
    nets = []
    for s in seeds:
        p = exp / "checkpoints" / "paper" / f"sts_extreme_{name}_s{s}_s{s}.pt"
        c = torch.load(p, map_location="cpu")
        if "inputs" in opt or ("pad" in opt and "focus" not in opt):
            from wildfire.paper.sts_inputs import build_masked
            sts_extreme.build = build_masked(opt.get("inputs", "all"), opt.get("pad", "zeros"))
        elif "focus" in opt:
            v = tuple(opt["focus"].split(","))
            coef = load_coef(exp / "derived_data" / "sts_focus" / f"coef_{name}_s{s}.npz") if "resid" in v else None
            sts_extreme.build = build_focus(v, coef, mu_f, sd_f, pad=opt.get("pad", "zeros"))
        try:
            net, _ = sts_extreme.load_any(str(p), c["init_px"], c["init_pa"], device)
        finally:
            sts_extreme.build = _ORIG_BUILD
        nets.append(net)
    return name, nets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="Rauchtest: so viele Test-Zeilen")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"focus_eval{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    mu_f, sd_f = float(mean[FIRE_CH]), float(std[FIRE_CH])
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    norm = lambda x: (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() - m_t) / s_t  # noqa: E731
    t0 = time.time()

    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    LC = Xp.shape[1]
    te = np.where(np.load(cd / "split.npy") == "test")[0]
    if args.limit:
        te = te[:args.limit]
    y = {"patch": targets.labels(cd, "next_patch")[te],
         "centre": targets.fire_map(cd, Xp, "next_patch")[te][:, CENTRE, CENTRE].astype(int)}
    doyD = doy_of(np.load(cd / "meta.npy")[:, 0])[te]
    Xt = np.asarray(Xp[te])
    loc = np.arange(te.size)
    sub = {}
    for dt_ in DTS:
        e = LC - 1 - dt_
        anyfire = (Xt[:, e - 29:e + 1, FIRE_CH] > 0).reshape(te.size, -1).any(1)
        sub[dt_] = {"all": np.ones(te.size, bool), "ign30": ~anyfire, "fire30": anyfire}
    print(f"[fev] Test {te.size:,}; ign30 je dt " + " ".join(f"{d}: {sub[d]['ign30'].sum():,}" for d in DTS)
          + f" ({time.time() - t0:.0f} s)", flush=True)

    scores = {}                                               # name -> tg -> (seeds, len(DTS), n)
    for spec in args.models:
        name, nets = load_nets(exp, spec, args.seeds, device, mu_f, sd_f)
        Fc = np.zeros((len(nets), len(DTS), te.size), np.float32)
        Fp = np.zeros_like(Fc)
        t1 = time.time()
        with torch.no_grad():
            for k, dt_ in enumerate(DTS):
                e = LC - 1 - dt_
                tau = torch.tensor([float(dt_ + 1)], device=device)
                for a in range(0, te.size, args.bs):
                    j = slice(a, a + args.bs)
                    x = norm(windows(Xt, loc[j], np.full(loc[j].size, e)))
                    d = torch.from_numpy((doyD[j] - dt_).astype(np.float32)).to(device)
                    for s, net in enumerate(nets):
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            raw_px, raw_pa = net(x, d)[:2]
                        Fc[s, k, j] = (-torch.expm1(net.px_head.log_survival(raw_px[:, CENTRE, CENTRE].float(), tau))[:, 0]).cpu().numpy()
                        Fp[s, k, j] = (-torch.expm1(net.pa_head.log_survival(raw_pa.float(), tau))[:, 0]).cpu().numpy()
        scores[name] = {"centre": Fc, "patch": Fp}
        print(f"[fev] {spec}: {time.time() - t1:.0f} s", flush=True)
        del nets
        torch.cuda.empty_cache()

    # ---- logistische Feuerhistorie-Basis
    pdir = exp / "derived_data" / args.cube_dir
    Xtr = np.load(pdir / "train_X.npy", mmap_mode="r")
    ftr = fire_of(Xtr, np.arange(Xtr.shape[0]))
    dates = CubeReader(exp).dates
    doy_tr = np.array([dt.date.fromisoformat(dates[int(L)][:10]).timetuple().tm_yday
                       for L in np.load(pdir / "train_meta.npz")["L"]], np.float32)
    feat = lambda f, d: np.concatenate([features_px(torch.from_numpy(f[a:a + 8000]), torch.from_numpy(d[a:a + 8000]))  # noqa: E731
                                        [:, CENTRE, CENTRE].numpy() for a in range(0, len(f), 8000)])
    Ftr = feat(ftr, doy_tr)
    del ftr
    ev = np.asarray(np.load(pdir / "train_event.npy", mmap_mode="r"))
    tt = np.asarray(np.load(pdir / "train_t.npy", mmap_mode="r")).astype(np.int64)
    pe, pt = patch_events(ev, tt)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Bc, Bp = np.zeros((1, len(DTS), te.size), np.float32), np.zeros((1, len(DTS), te.size), np.float32)
    for k, dt_ in enumerate(DTS):
        e = LC - 1 - dt_
        Fte = feat(Xt[:, e - 29:e + 1, FIRE_CH] > 0, (doyD - dt_).astype(np.float32))
        for B_, yy in ((Bc, ev[:, CENTRE, CENTRE] & (tt[:, CENTRE, CENTRE] <= dt_ + 1)), (Bp, pe & (pt <= dt_ + 1))):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Ftr, yy.astype(int))
            B_[0, k] = m.predict_proba(Fte)[:, 1]
    scores["fire_logit"] = {"centre": Bc, "patch": Bp}
    print(f"[fev] logistische Basis ({time.time() - t0:.0f} s)", flush=True)

    # ---- U-Net-Nachbau
    rp = exp / "predictions" / "paper" / "rec_cache_G_cache_next_paperhp_w0.npz"
    if rp.exists():
        z = np.load(rp)
        pos = {int(i): n for n, i in enumerate(z["idx_test"])}
        if all(int(i) in pos for i in te):
            sel = np.array([pos[int(i)] for i in te])
            U = np.stack([np.stack([z[f"paper55_T30_end{d}_s{s}_test"][sel] for d in DTS]) for s in range(3)
                          if f"paper55_T30_end1_s{s}_test" in z.files]).astype(np.float32)
            scores["unet_rec"] = {"patch": U}
        else:
            print("[fev] U-Net: Test-Zeilen fehlen - uebersprungen", flush=True)

    # ---- Metriken
    res = {}
    for name, d in scores.items():
        for tg, F_ in d.items():
            for k, dt_ in enumerate(DTS):
                for sn, m in sub[dt_].items():
                    yy = y[tg][m]
                    if yy.sum() == 0:
                        continue
                    per = [float(average_precision_score(yy, F_[s, k, m])) for s in range(F_.shape[0])]
                    res.setdefault(name, {}).setdefault(tg, {}).setdefault(sn, {})[str(dt_)] = {
                        "pr_auc_mean": float(np.mean(per)), "pr_auc_sd": float(np.std(per)),
                        "pr_auc_ens": float(average_precision_score(yy, F_[:, k, m].mean(0))),
                        "n": int(m.sum()), "pos": int(yy.sum()), "base": float(yy.mean())}
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "scores.npz", rows=te, **{f"{n}__{tg}": F_ for n, d in scores.items() for tg, F_ in d.items()})
    md = ["# Paper-Protokoll je dt: alle Test-Stichproben und neue Zuendungen (ign30)", "",
          "PR-AUC, Mittel ueber die Seeds (Ensemble in Klammern). ign30 = kein Feuer im Ausschnitt an den 30 Fenstertagen.",
          f"U-Net-Nachbau laut Referenz auf allen: {' / '.join(f'{UNET_ALL[d]:.3f}' for d in DTS)}.", ""]
    for tg, lab in (("patch", "Ausschnitt"), ("centre", "Mitte")):
        for sn in ("all", "ign30", "fire30"):
            r0 = next(iter(res.values()))[tg][sn] if tg in next(iter(res.values())) else None
            info = "" if r0 is None else " · ".join(f"dt {d}: n {r0[str(d)]['n']:,}, pos {r0[str(d)]['pos']:,} "
                                                    f"(Basisrate {r0[str(d)]['base']:.3f})" for d in DTS)
            md += [f"## {lab}, {sn}", "", info, "", "| Modell | " + " | ".join(f"dt {d}" for d in DTS) + " |",
                   "|---" * (1 + len(DTS)) + "|"]
            for name, r in res.items():
                if tg not in r or sn not in r[tg]:
                    continue
                md.append(f"| {name} | " + " | ".join(
                    f"{r[tg][sn][str(d)]['pr_auc_mean']:.3f} ({r[tg][sn][str(d)]['pr_auc_ens']:.3f})"
                    if str(d) in r[tg][sn] else "–" for d in DTS) + " |")
            md.append("")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p45_focus_eval{args.tag}", args.config, {
        "models": args.models, "seeds": args.seeds, "n_test": int(te.size),
        "subsets": {str(d): {k: int(v.sum()) for k, v in s.items()} for d, s in sub.items()},
        "baseline": "LogisticRegression auf sts_focus.features_px (Mittelpixel), je tau = dt + 1, Wuerfel-Stichtage",
        "results": res, "out": str(out)})


if __name__ == "__main__":
    main()
