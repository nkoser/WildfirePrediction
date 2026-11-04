"""Gleiche Nachkalibrierung fuer alle Modelle, gefittet auf den Val-Jahren (2020-2022, surv_eval_test/val_*).

Alle Modelle sind auf dem Wuerfel trainiert, der rund elfmal mehr Feuer enthaelt als zufaellige Stichtage. Mit dem
Trainings-Bias sagen sie zu viel Feuer voraus; der Standard-Bias von load_sts gleicht das zufaellig teilweise aus
(siehe joint_eval --bias). Hier bekommt jedes Modell stattdessen zwei Skalare auf der Logit-Skala des Tages-Hazards,
ausgehend vom Trainings-Bias, gefittet durch die Summe zweier mittlerer NLL auf den Val-Stichtagen:

    Pixel       J zufaellige Pixel je Stichtag (erstes Feuer im Pixel)
    Ausschnitt  erstes Feuer irgendwo im 13 x 13-Ausschnitt

  lh  STS-LH (auch lh_region):  d_px auf px_head.bias, d_pa auf pa_head.bias (getrennt fittbar, je ein Gitter)
  cg  Contagion-STS:            d_a auf ign_bias (Zuendung), d_k auf die Logits des Kerns ausser der Mitte
                                (Ausbreitung; die Mitte ist die Persistenz beobachteter Feuer). Pixel ueber
                                Rao-Blackwell-Monte-Carlo (mc_pixel, gemeinsame Zufallszahlen fuer alle
                                Gitterpunkte), Ausschnitt exakt. Suche: d_a-Linie bei d_k = 0, dann Mustersuche.

Die Regionskoepfe von lh_region bleiben unveraendert. Die Val-Jahre haben weniger Feuer als die Testjahre
(Mitte 0,28 % gegen 0,77 % bis 100 Tage); die Kalibrierung gleicht den Wuerfel aus, nicht den Jahresunterschied.

  --part events     Ereignisse je Pixel fuer die Val-Stichtage -> derived_data/joint_calib/val_pixel_events.npz
  --part fit        je Checkpoint -> metrics/paper_joint/<calib-dir>/<Checkpoint>.json
  --part shapes     joint_eval --part shapes bzw. scenarios mit den kalibrierten Modellen (Ausgaben wie dort,
  --part scenarios  eigenes --tag); load_model wird dafuer nur in diesem Prozess ersetzt.

Alle Eingaben nur gelesen; bestehende Ausgaben werden nicht ueberschrieben (Abbruch).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config
from wildfire.paper import joint_eval
from wildfire.paper.joint_eval import lh_parts, parse_models
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.region_eval import H, HALF, N

EV_DIR = "joint_calib"
_load_trained = joint_eval.load_model                      # Original; run_eval ersetzt joint_eval.load_model


# ---------------------------------------------------------------------------- Bausteine (auch fuer joint_cond)
def nll_curve(ls: torch.Tensor, e: torch.Tensor, t: torch.Tensor, k0: torch.Tensor | None = None) -> torch.Tensor:
    """NLL je Zeile aus log S (R, H+1), Ereignis e, Tag t (1..H); mit k0 die Restzeit nach Tag k0 (t > k0)."""
    t = t.long().clamp(1, H)
    prev = ls.gather(1, (t - 1)[:, None])[:, 0]
    cur = ls.gather(1, t[:, None])[:, 0]
    base = ls.gather(1, k0.long()[:, None])[:, 0] if k0 is not None else torch.zeros_like(prev)
    ev = -(prev - base + torch.log(torch.clamp(-torch.expm1(cur - prev), min=1e-15)))
    return torch.where(e, ev, -(ls[:, H] - base))


@torch.no_grad()
def mc_pixel(net, log_exo, k, logD, land, pix, gen, start=None, trig=None):
    """Rao-Blackwell-Monte-Carlo je Zeile: log prod (1 - h_p) (R, H+1) fuer das Pixel p = pix[r], das in seiner Zeile
    nicht zuendet (wie mc_region_logS mit R = {p}); land[r] = Stichtag der Zeile. Die Zeilen enthalten die Pfade
    schon (je Pfad eine Zeile), gemittelt wird danach.

    start (R,) Tag k: bis Tag k brennt nichts, nur trig (R, N) zuendet an Tag k; danach frei. Summiert wird ab Tag
    k + 1, Kurve[k + r] ist dann log S der Restzeit r. Ohne start ab Tag 1 wie im Stichtag."""
    R = land.numel()
    Hh, A = log_exo.shape[1], logD.shape[-1]
    c = net.K * net.K // 2
    dev = log_exo.device
    rows = torch.arange(R, device=dev)
    kk = k[land]
    sup = torch.zeros(R, N, dtype=torch.bool, device=dev)
    sup[rows, pix] = True
    fired = torch.full((R, N), -1, dtype=torch.long, device=dev)
    cum = torch.zeros(R, dtype=torch.float64, device=dev)
    curve = [cum]
    cols = torch.arange(N, device=dev)[None]
    for s in range(Hh):
        on = fired >= 0
        age = (s - fired - 1).clamp(min=0, max=A - 1)
        b = torch.where(on, logD[land[:, None], cols, age].exp(), torch.zeros(R, N, device=dev))
        lp = torch.log1p(-(kk * net._unfold(b)).clamp(max=1 - 1e-6))
        l1mh = log_exo[land, s] + lp.sum(1) - lp[:, c]
        act = torch.ones(R, dtype=torch.bool, device=dev) if start is None else s >= start
        cum = cum + torch.where(act, l1mh[rows, pix].double(), torch.zeros_like(cum))
        curve.append(cum)
        u = torch.rand((R, N), generator=gen, device=dev)
        new = (~on) & (~sup) & act[:, None] & (u < -torch.expm1(l1mh))
        if trig is not None:
            new = new | (trig & (start - 1 == s)[:, None])
        fired = torch.where(new, torch.full_like(fired, s), fired)
    return torch.stack(curve, 1)


def path_mean(curves: torch.Tensor, M: int) -> torch.Tensor:
    """(R*M, H+1) log prod je Pfad -> (R, H+1) log S, Pfade innen."""
    return torch.logsumexp(curves.view(-1, M, curves.shape[-1]), 1) - np.log(M)


def noncentre(net, device):
    m = torch.ones(net.K * net.K, device=device)
    m[net.K * net.K // 2] = 0
    return m


def apply_calib(net, cal: dict) -> None:
    d = cal["delta"]
    with torch.no_grad():
        if getattr(net, "head_kind", "") == "contagion":
            net.ign_bias = net.ign_bias + d["a"]
            net.ker.bias.add_(d["k"] * noncentre(net, net.ker.bias.device))
        else:
            net.px_head.bias = net.px_head.bias + d["px"]
            net.pa_head.bias = net.pa_head.bias + d["pa"]


def calib_file(exp, calib_dir: str, ck) -> Path:
    return exp / "metrics" / "paper_joint" / calib_dir / f"{Path(ck).stem}.json"


def load_calibrated(exp, calib_dir: str, ck, device, cube_rates):
    """Trainings-Bias wie joint_eval --bias trained, dann die Skalare aus <calib_dir>."""
    f = calib_file(exp, calib_dir, ck)
    if not f.exists():
        raise SystemExit(f"ABBRUCH: keine Kalibrierung {f}")
    cal = json.loads(f.read_text())
    net, used = _load_trained(ck, device, "trained", cube_rates)
    apply_calib(net, cal)
    return net, {"rates": list(used), "delta": cal["delta"]}


def norm_tensors(cfg, args, device):
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    return m_t, s_t


def cube_rates_of(exp, args):
    ev_tr = np.load(exp / "derived_data" / args.cube_dir / "train_event.npy").astype(bool)
    return float(ev_tr.mean()), float(ev_tr.reshape(len(ev_tr), -1).any(1).mean())


# ---------------------------------------------------------------------------- Ereignisse der Val-Stichtage
def build_events(cfg, exp, args):
    from wildfire.data.dataset import CubeReader
    from wildfire.paper.pixel_landmarks import pixel_events
    from wildfire.paper.surv_eval import raw_fire_keys

    out = exp / "derived_data" / EV_DIR / "val_pixel_events.npz"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - wird nicht ueberschrieben")
    me = dict(np.load(exp / "derived_data" / args.surv_dir / "val_meta.npz"))
    cube = CubeReader(exp)
    keys, _ = raw_fire_keys(Path(cfg["dataset_root"]))
    ev, t = pixel_events(keys, me["L_abs"], me["row"], me["col"], cube.H, cube.W)
    pe, pt = ev.reshape(len(ev), -1).any(1), t.reshape(len(t), -1).min(1)
    checks = {"ev_patch": float((pe == me["ev_patch"].astype(bool)).mean()),
              "t_patch": float((pt[pe] == me["t_patch"][pe]).mean()),
              "ev_centre": float((ev[:, HALF, HALF] == me["ev_centre"].astype(bool)).mean())}
    print(f"[calib] Selbsttest Ereignisse je Pixel gegen val_meta: {checks}", flush=True)
    if min(checks.values()) < 0.9999:
        raise SystemExit("ABBRUCH: Ereignisse je Pixel passen nicht zu val_meta.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ev=ev, t=t, row=me["row"], col=me["col"], L_abs=me["L_abs"])
    print(f"[calib] {out}: {ev.shape}, Pixel-Ereignisrate {ev.mean():.4f}", flush=True)


# ---------------------------------------------------------------------------- Fit
def val_setup(exp, args):
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "val_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / EV_DIR / "val_pixel_events.npz"))
    assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Val-Stichtage passen nicht"
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(me["row"]), size=min(args.n_val, len(me["row"])), replace=False))
    picks = np.stack([rng.choice(N, size=args.J, replace=False) for _ in idx])                   # (n, J)
    t0 = time.time()
    X = torch.from_numpy(np.ascontiguousarray(np.load(sd / "val_X.npy", mmap_mode="r")[idx]))
    evp = pe["ev"].reshape(len(me["row"]), N)[idx]
    tp = pe["t"].reshape(len(me["row"]), N)[idx].astype(np.int64)
    rows = np.arange(len(idx))[:, None]
    tg = {"px_e": evp[rows, picks].ravel(), "px_t": tp[rows, picks].ravel(),
          "pa_e": me["ev_patch"][idx].astype(bool), "pa_t": np.clip(me["t_patch"][idx].astype(np.int64), 1, H)}
    print(f"[calib] {len(idx):,} Val-Stichtage, {args.J} Pixel je Stichtag, geladen in {time.time() - t0:.0f} s; "
          f"Ereignisse Pixel {tg['px_e'].sum():,} ({tg['px_e'].mean():.4f}), Ausschnitt {tg['pa_e'].sum():,} "
          f"({tg['pa_e'].mean():.4f})", flush=True)
    return X, me["doy"][idx].astype(np.float32), picks, tg


@torch.no_grad()
def fit_lh(net, X, doy, picks, tg, m_t, s_t, device, args):
    for hd in (net.px_head, net.pa_head):                     # log(1 - h) = logsigmoid(-(roh + bias)) unten
        assert type(hd).__name__ == "LogisticHazardHead", type(hd).__name__
    rp, ra = [], []
    for a0 in range(0, X.shape[0], args.bs):
        x = (X[a0:a0 + args.bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a0:a0 + args.bs]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, raw_px, raw_pa = lh_parts(net, x, d)
        pk = torch.from_numpy(picks[a0:a0 + args.bs]).to(device)
        rp.append(raw_px.float().gather(1, pk[..., None].expand(-1, -1, raw_px.shape[-1])).reshape(-1, H))
        ra.append(raw_pa.float())
    raw = {"px": torch.cat(rp).double(), "pa": torch.cat(ra).double()}
    bias = {"px": float(net.px_head.bias), "pa": float(net.pa_head.bias)}
    tgt = {h: (torch.from_numpy(tg[f"{h}_e"]).to(device), torch.from_numpy(tg[f"{h}_t"]).to(device)) for h in raw}

    def nll(h, dl):
        ls = F.pad(F.logsigmoid(-(raw[h] + bias[h] + dl)).cumsum(1), (1, 0))
        return float(nll_curve(ls, *tgt[h]).mean())

    grid = np.round(np.arange(args.lh_lo, args.lh_hi + 1e-9, args.lh_step), 4)
    out, trace = {}, {}
    for h in raw:
        v = [nll(h, float(g)) for g in grid]
        j = int(np.argmin(v))
        if j in (0, len(grid) - 1):
            print(f"[calib] WARNUNG: {h} Minimum am Rand des Gitters ({grid[j]})", flush=True)
        out[h] = float(grid[j])
        trace[h] = {"grid": grid.tolist(), "nll": v}
    before = {h: nll(h, 0.0) for h in raw}
    after = {h: nll(h, out[h]) for h in raw}
    return out, before, after, trace


@torch.no_grad()
def fit_cg(net, X, doy, picks, tg, m_t, s_t, device, args):
    keep = {"obs": [], "ign": [], "ker": [], "logD": []}
    for a0 in range(0, X.shape[0], args.bs):
        x = (X[a0:a0 + args.bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a0:a0 + args.bs]).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            xx = x[:, -net.hist:]
            pix = net.encode_input(xx, d).float()
            obs, _, _, logD = net.params(x, d, pix)
            keep["obs"].append(obs)
            keep["ign"].append((net.ign(pix) + net.ign_bias).transpose(1, 2))
            keep["ker"].append(net.ker(pix).transpose(1, 2))
            keep["logD"].append(logD)
    V = {k_: torch.cat(v) for k_, v in keep.items()}
    nc = noncentre(net, device)[None, :, None]
    n, J, M = X.shape[0], args.J, args.M
    pk = torch.from_numpy(picks).to(device)
    px_e, px_t = torch.from_numpy(tg["px_e"]).to(device), torch.from_numpy(tg["px_t"]).to(device)
    pa_e, pa_t = torch.from_numpy(tg["pa_e"]).to(device), torch.from_numpy(tg["pa_t"]).to(device)
    lpc = max(1, args.rows // (J * M))                       # Stichtage je Block
    cache = dict(getattr(args, "seed_cache", {}))            # Punkte aus --resume-log (gleiche CRN, also gleich)
    if cache:
        print(f"[calib]   {len(cache)} Punkte aus frueherem Lauf uebernommen", flush=True)

    def objective(da: float, dk: float):
        key = (round(da, 4), round(dk, 4))
        if key in cache:
            return cache[key]
        t0 = time.time()
        s_px = s_pa = 0.0
        for bi, a0 in enumerate(range(0, n, lpc)):
            sl = slice(a0, min(n, a0 + lpc))
            a = torch.sigmoid(V["ign"][sl] + da)
            k = torch.sigmoid(V["ker"][sl] + dk * nc)
            log_exo = net.exogenous(V["obs"][sl], a, k, V["logD"][sl])
            ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0)).double()
            s_pa += float(nll_curve(ls_pa, pa_e[sl], pa_t[sl]).sum())
            b = log_exo.shape[0]
            land = torch.arange(b, device=device).repeat_interleave(J * M)
            pix = pk[sl].repeat_interleave(M, 1).reshape(-1)
            gen = torch.Generator(device=device).manual_seed(args.seed * 100_003 + bi)   # gemeinsame Zufallszahlen
            ls_px = path_mean(mc_pixel(net, log_exo, k, V["logD"][sl], land, pix, gen), M)
            r0 = a0 * J
            s_px += float(nll_curve(ls_px, px_e[r0:r0 + b * J], px_t[r0:r0 + b * J]).sum())
        v = (s_px / (n * J), s_pa / n)
        cache[key] = v
        print(f"[calib]   d_a {da:+.3f}  d_k {dk:+.3f}:  Pixel {v[0]:.5f}  Ausschnitt {v[1]:.5f}  Summe {sum(v):.5f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
        return v

    f = lambda da, dk: sum(objective(da, dk))  # noqa: E731
    line = [float(v) for v in args.cg_line]
    best = (min(line, key=lambda da: f(da, 0.0)), 0.0)
    step = args.cg_step
    while step >= args.cg_min_step - 1e-9:                    # Mustersuche
        moved = False
        for dda, ddk in ((step, 0), (-step, 0), (0, step), (0, -step)):
            cand = (best[0] + dda, best[1] + ddk)
            if f(*cand) < f(*best) - 1e-7:
                best, moved = cand, True
                break
        if not moved:
            step /= 2
    out = {"a": float(best[0]), "k": float(best[1])}
    v0, v1 = objective(0.0, 0.0), objective(*best)
    trace = [{"a": k_[0], "k": k_[1], "px": v[0], "pa": v[1]} for k_, v in cache.items()]
    return out, {"px": v0[0], "pa": v0[1]}, {"px": v1[0], "pa": v1[1]}, trace


def parse_resume_logs(paths):
    """Bereits berechnete Zielwerte aus Fit-Logs: {ckpt-stem: {(d_a, d_k): (px, pa)}} (5 Nachkommastellen)."""
    import re
    head = re.compile(r"^\[calib\] \S+ (\S+) \((?:cg|lh)\), Trainings-Bias")
    pt = re.compile(r"d_a ([+-]\d+\.\d+)  d_k ([+-]\d+\.\d+):  Pixel (\d+\.\d+)  Ausschnitt (\d+\.\d+)")
    out, stem = {}, None
    for p in paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            h = head.match(line)
            if h:
                stem = h.group(1)
                continue
            m = pt.search(line)
            if m and stem is not None:
                key = (round(float(m.group(1)), 4), round(float(m.group(2)), 4))
                out.setdefault(stem, {})[key] = (float(m.group(3)), float(m.group(4)))
    return out


def run_fit(cfg, exp, args, device):
    m_t, s_t = norm_tensors(cfg, args, device)
    rates = cube_rates_of(exp, args)
    X, doy, picks, tg = val_setup(exp, args)
    resumed = parse_resume_logs(args.resume_log)
    for name, kind, cks in parse_models(args.models):
        for ck in cks:
            out = calib_file(exp, args.calib_dir, ck)
            if out.exists():
                if args.skip_existing:
                    print(f"[calib] {out} existiert schon, uebersprungen", flush=True)
                    continue
                raise SystemExit(f"ABBRUCH: {out} existiert schon")
            args.seed_cache = resumed.get(Path(ck).stem, {}) if kind == "cg" else {}
            net, used = _load_trained(ck, device, "trained", rates)
            t0 = time.time()
            print(f"[calib] {name} {Path(ck).stem} ({kind}), Trainings-Bias aus Raten {used}", flush=True)
            fit = fit_cg if kind == "cg" else fit_lh
            delta, before, after, trace = fit(net, X, doy, picks, tg, m_t, s_t, device, args)
            print(f"[calib] {Path(ck).stem}: delta {delta}; Val-NLL vorher {before} -> nachher {after} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"ckpt": str(ck), "name": name, "kind": kind, "delta": delta,
                                       "val_before": before, "val_after": after, "trained_rates": list(used),
                                       "n_val": int(X.shape[0]), "J": args.J, "M": args.M, "seed": args.seed,
                                       "trace": trace}, indent=1))
            print(f"[calib] {out}", flush=True)


# ---------------------------------------------------------------------------- Auswertung mit Kalibrierung
def run_eval(cfg, exp, args, device):
    def load(ck, device_, mode, cube_rates):
        return load_calibrated(exp, args.calib_dir, ck, device_, cube_rates)
    joint_eval.load_model = load                              # nur in diesem Prozess
    args.bias = f"calib:{args.calib_dir}"
    (joint_eval.run_shapes if args.part == "shapes" else joint_eval.run_scenarios)(cfg, exp, args, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["events", "fit", "shapes", "scenarios"], required=True)
    ap.add_argument("--models", nargs="+", default=[], help="name=cg|lh:glob")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--n-val", type=int, default=10_000)
    ap.add_argument("--J", type=int, default=16, help="fit: Pixel je Val-Stichtag")
    ap.add_argument("--M", type=int, default=16, help="fit: Pfade; shapes/scenarios: wie joint_eval")
    ap.add_argument("--rows", type=int, default=65_536, help="fit cg: Zeilen (Stichtag x Pixel x Pfad) je Block")
    ap.add_argument("--lh-lo", type=float, default=-5.0)
    ap.add_argument("--lh-hi", type=float, default=2.0)
    ap.add_argument("--lh-step", type=float, default=0.05)
    ap.add_argument("--cg-line", nargs="+", default=["-4", "-3", "-2", "-1", "0"])
    ap.add_argument("--cg-step", type=float, default=0.5)
    ap.add_argument("--cg-min-step", type=float, default=0.125)
    ap.add_argument("--resume-log", nargs="*", default=[], help="fit cg: Logs frueherer Laeufe, Punkte uebernehmen")
    ap.add_argument("--skip-existing", action="store_true", help="fit: vorhandene Kalibrierungen ueberspringen")
    ap.add_argument("--n", type=int, default=20_000, help="shapes/scenarios: Test-Stichtage wie joint_eval")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.part == "events":
        build_events(cfg, exp, args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.part == "fit":
        run_fit(cfg, exp, args, device)
    else:
        if not args.tag:
            raise SystemExit("ABBRUCH: --tag noetig")
        run_eval(cfg, exp, args, device)


if __name__ == "__main__":
    main()
