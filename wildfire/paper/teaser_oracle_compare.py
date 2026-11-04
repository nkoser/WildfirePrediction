"""Teaser-Ereignisse: STS-LH, Contagion-STS und STS-LH mit Wetter-Oracle an denselben Pixeln und Ausschnitten.

Liest die npz von risk_map / risk_map_oracle (risk_<tag>_<datum>.npz). Die Beispielpixel a/b/c werden wie in
teaser_spread.pick_pixels auf der Karte des Bezugsmodells (--ref, Standard Contagion-STS) gewaehlt und fuer
alle Modelle an genau diesen Pixeln ausgelesen (in den Einzelabbildungen waehlt jedes Modell eigene Pixel).
Dazu je Ausschnitt teaser_spread.box_stats und der Abstand jedes Beispielpixels zum naechsten Feuerpixel im
Eingabefenster.

Ausgabe: metrics/teaser_oracle<tag>/summary.md und summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from wildfire.audit.common import load_config
from wildfire.paper.risk_map_plot import grid_geo
from wildfire.paper.teaser_spread import arrival_pred, box_slice, box_stats, load, pick_pixels

TAUS_SHOW = (1, 3, 7, 8, 14, 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tags", default="teaser_lh,teaser_cgs,teaser_lho")
    ap.add_argument("--names", default="STS-LH,Contagion-STS,STS-LH + weather oracle")
    ap.add_argument("--ref", default="teaser_cgs")
    ap.add_argument("--events", default="/data/Wildfire/figures/teaser/teaser_v1_cgs.json")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    geo = grid_geo(cfg)
    tags, names = args.tags.split(","), args.names.split(",")
    events = [(d, tuple(b)) for d, b in json.loads(Path(args.events).read_text())["events"]]
    out_dir = exp / "metrics" / f"teaser_oracle{args.tag}"
    if out_dir.exists():
        raise SystemExit(f"ABBRUCH: {out_dir} existiert")
    res, md = {}, ["# Teaser-Ereignisse: STS-LH, Contagion-STS, Wetter-Oracle", "",
                   f"Beispielpixel gewaehlt auf {args.ref} (teaser_spread.pick_pixels), Schwelle p = {args.p}. "
                   "Rohe, nicht kalibrierte F(tau). Abstand = euklidisch in Pixeln zum naechsten Feuerpixel im Eingabefenster.", ""]
    for date, box in events:
        ref = load(exp, args.ref, date)
        sl = box_slice(box, geo, ref["mask"].shape)
        pix = pick_pixels(ref, sl, args.p)
        dist = distance_transform_edt(~ref["win"][sl])
        obs = np.where(ref["first"][sl] > 0, ref["first"][sl], 31)
        e = res[f"{date}:{list(box)}"] = {"pixels": {}, "box": {}}
        md += [f"## {date}, Ausschnitt {box}", "",
               "| Pixel | beob. Tag | Abstand px | Modell | " + " | ".join(f"F({t})" for t in TAUS_SHOW) + f" | Ankunft p={args.p} |",
               "|---|---|---|---|" + "---|" * len(TAUS_SHOW) + "---|"]
        for k, (grp, r, c) in enumerate(pix):
            lab = "abc"[k]
            e["pixels"][lab] = {"group": grp, "row": int(r + sl[0].start), "col": int(c + sl[1].start),
                                "observed_day": int(obs[r, c]), "distance_px": float(dist[r, c]), "F": {}}
            for tag, name in zip(tags, names):
                d = load(exp, tag, date)
                Fx = d["F"][sl][r, c]
                a = int(arrival_pred(Fx[None], args.p)[0])
                e["pixels"][lab]["F"][name] = {str(t): float(Fx[t - 1]) for t in range(1, Fx.shape[-1] + 1)}
                md.append(f"| {lab} ({grp}) | {obs[r, c] if obs[r, c] <= 30 else '–'} | {dist[r, c]:.1f} | {name} | "
                          + " | ".join(f"{Fx[t - 1]:.3f}" for t in TAUS_SHOW) + f" | {a if a <= 30 else '–'} |")
        md += ["", "| Modell | " + " | ".join(f"AUC {t} d" for t in (1, 7, 14, 30))
               + f" | Recall 7 d | Praez. 7 d | Recall 30 d | Praez. 30 d |", "|---|" + "---|" * 8]
        for tag, name in zip(tags, names):
            d = load(exp, tag, date)
            st = box_stats(d, sl, [args.p])
            e["box"][name] = st
            g = lambda k: f"{st[k]:.3f}" if k in st else "–"
            md.append(f"| {name} | " + " | ".join(g(f"auc_{t}") for t in (1, 7, 14, 30)) + " | "
                      + " | ".join(g(f"p{args.p}_t{t}_{m}") for t in (7, 30) for m in ("recall", "precision")) + " |")
        md.append("")
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
