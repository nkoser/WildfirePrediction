"""focus_eval mit waehlbarem Paper-Ziel: D+1 (wie focus_eval) oder D (nur Auswertung, dieselben Checkpoints).

Die Netze sind auf Wuerfel-Stichtagen trainiert, das Training haengt nicht vom Paper-Ziel ab (wie sts_inputs_target).
  next_patch  Zieltag D+1: Eingabe endet bei D - dt (Index LC-1-dt), Score F(dt + 1), Label y_next_patch / Mitte D+1
              -> wiederholt focus_eval exakt (Selbsttest)
  cache       Zieltag D:   Eingabe endet bei D - dt - 1 (Index LC-2-dt), Score F(dt + 1), Label y (Cache) / Mitte D
Teilmengen je dt wie focus_eval (all, ign30, fire30), bezogen auf die 30 Fenstertage vor dem jeweiligen Eingabeende.
Logistische Feuerhistorie wie focus_eval (tau = dt + 1, Merkmale am Eingabeende). U-Net-Nachbau aus
predictions/paper/rec_<cache>_cache<run>.npz, run = _next_paperhp_w0 (D+1, end = dt) bzw. _paperhp (D, end = dt + 1).
Modelle wie focus_eval (name, name:focus=..., name:inputs=...).
Ausgabe NEU: metrics/focus_eval_target<tag>/ (summary.md, results.json), Manifest p52_focus_eval_target<tag>.
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
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.fire_baseline import fire_of
from wildfire.paper.focus_eval import load_nets
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import CENTRE, doy_of, patch_events
from wildfire.paper.sts_focus import features_px

UNET = {"next_patch": ("_next_paperhp_w0", 0), "cache": ("_paperhp", 1)}     # run-name, end - dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--target", choices=["next_patch", "cache"], required=True)
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
    out = exp / "metrics" / f"focus_eval_target{args.tag}"
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
    sh = targets.shift(args.target)                                     # 1 fuer D+1, 0 fuer D
    end = lambda d: LC - 1 - d - (1 - sh)  # noqa: E731                 # Index des letzten Eingabetags
    te = np.where(np.load(cd / "split.npy") == "test")[0]
    if args.limit:
        te = te[:args.limit]
    y = {"patch": targets.labels(cd, args.target)[te],
         "centre": targets.fire_map(cd, Xp, args.target)[te][:, CENTRE, CENTRE].astype(int)}
    doyD = doy_of(np.load(cd / "meta.npy")[:, 0])[te]
    Xt = np.asarray(Xp[te])
    loc = np.arange(te.size)
    sub = {}
    for dt_ in DTS:
        e = end(dt_)
        anyfire = (Xt[:, e - 29:e + 1, FIRE_CH] > 0).reshape(te.size, -1).any(1)
        sub[dt_] = {"all": np.ones(te.size, bool), "ign30": ~anyfire, "fire30": anyfire}
    print(f"[fet] Ziel {args.target}, Test {te.size:,}; Basisrate Ausschnitt {y['patch'].mean():.3f}; ign30 je dt "
          + " ".join(f"{d}: {sub[d]['ign30'].sum():,}" for d in DTS) + f" ({time.time() - t0:.0f} s)", flush=True)

    scores = {}
    for spec in args.models:
        name, nets = load_nets(exp, spec, args.seeds, device, mu_f, sd_f)
        Fc = np.zeros((len(nets), len(DTS), te.size), np.float32)
        Fp = np.zeros_like(Fc)
        t1 = time.time()
        with torch.no_grad():
            for k, dt_ in enumerate(DTS):
                e = end(dt_)
                tau = torch.tensor([float(dt_ + 1)], device=device)
                # Kalendertag des letzten Eingabetags: D - dt (D+1) bzw. D - dt - 1 (D)
                for a in range(0, te.size, args.bs):
                    j = slice(a, a + args.bs)
                    x = norm(windows(Xt, loc[j], np.full(loc[j].size, e)))
                    d = torch.from_numpy((doyD[j] - dt_ - (1 - sh)).astype(np.float32)).to(device)
                    for s, net in enumerate(nets):
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            raw_px, raw_pa = net(x, d)[:2]
                        Fc[s, k, j] = (-torch.expm1(net.px_head.log_survival(raw_px[:, CENTRE, CENTRE].float(), tau))[:, 0]).cpu().numpy()
                        Fp[s, k, j] = (-torch.expm1(net.pa_head.log_survival(raw_pa.float(), tau))[:, 0]).cpu().numpy()
        scores[name] = {"centre": Fc, "patch": Fp}
        print(f"[fet] {spec}: {time.time() - t1:.0f} s", flush=True)
        del nets
        torch.cuda.empty_cache()

    # ---- logistische Feuerhistorie-Basis (wie focus_eval)
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
        e = end(dt_)
        Fte = feat(Xt[:, e - 29:e + 1, FIRE_CH] > 0, (doyD - dt_ - (1 - sh)).astype(np.float32))
        for B_, yy in ((Bc, ev[:, CENTRE, CENTRE] & (tt[:, CENTRE, CENTRE] <= dt_ + 1)), (Bp, pe & (pt <= dt_ + 1))):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Ftr, yy.astype(int))
            B_[0, k] = m.predict_proba(Fte)[:, 1]
    scores["fire_logit"] = {"centre": Bc, "patch": Bp}
    print(f"[fet] logistische Basis ({time.time() - t0:.0f} s)", flush=True)

    # ---- U-Net-Nachbau
    run, off = UNET[args.target]
    rp = exp / "predictions" / "paper" / f"rec_{args.cache}_cache{run}.npz"
    if rp.exists():
        z = np.load(rp)
        pos = {int(i): n for n, i in enumerate(z["idx_test"])}
        if all(int(i) in pos for i in te):
            sel = np.array([pos[int(i)] for i in te])
            U = np.stack([np.stack([z[f"paper55_T30_end{d + off}_s{s}_test"][sel] for d in DTS]) for s in range(3)
                          if f"paper55_T30_end{1 + off}_s{s}_test" in z.files]).astype(np.float32)
            scores["unet_rec"] = {"patch": U}
        else:
            print("[fet] U-Net: Test-Zeilen fehlen - uebersprungen", flush=True)
    else:
        print(f"[fet] U-Net: {rp.name} fehlt - uebersprungen", flush=True)

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
    lab_t = "D+1" if args.target == "next_patch" else "D"
    md = [f"# Paper-Protokoll je dt, Ziel {lab_t}: alle Test-Stichproben und neue Zuendungen (ign30)", "",
          f"Cache {args.cache}, Wuerfel {args.cube_dir}. PR-AUC, Mittel ueber die Seeds (Ensemble in Klammern).", ""]
    for tg, lab in (("patch", "Ausschnitt"), ("centre", "Mitte")):
        for sn in ("all", "ign30", "fire30"):
            md += [f"## {lab}, {sn}", "", "| Modell | " + " | ".join(f"dt {d}" for d in DTS) + " |",
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
    write_manifest(cfg, f"p52_focus_eval_target{args.tag}", args.config, {
        "target": args.target, "models": args.models, "seeds": args.seeds, "cache": args.cache,
        "cube_dir": args.cube_dir, "n_test": int(te.size), "results": res, "out": str(out)})


if __name__ == "__main__":
    main()
