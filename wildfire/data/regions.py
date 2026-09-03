"""Rasterisiert die kanadischen Provinzen und Territorien auf das Gitter.

Fuer den raeumlichen OOD-Split brauchen wir Regionen, die eine fachliche
Bedeutung haben. Oekozonengrenzen liegen dem Datensatz nicht bei;
Provinzen sind die naechstbeste verteidigbare Einteilung und decken sich
in Kanada grob mit den Feuerregimen (Praerie, borealer Schild,
Kordillere, Atlantik).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize

from wildfire.audit.common import assert_readonly, load_config, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--provinces-geojson",
        default="/data/Wildfire/artifacts/ne_50m_admin_1_states_provinces.geojson",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    mask_dir = exp / "derived_data" / "masks"
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    H, W = man["grid"]["height"], man["grid"]["width"]
    tr = man["grid"]["transform"]
    transform = rasterio.Affine(*tr)

    gj = json.loads(Path(args.provinces_geojson).read_text())
    feats = [f for f in gj["features"] if f["properties"].get("admin") == "Canada"]
    names = sorted({(f["properties"].get("name") or "?") for f in feats})
    code_of = {n: i + 1 for i, n in enumerate(names)}   # 0 = ausserhalb

    shapes = []
    for f in feats:
        n = f["properties"].get("name") or "?"
        shapes.append((f["geometry"], code_of[n]))
    region = rasterize(shapes, out_shape=(H, W), transform=transform,
                       fill=0, all_touched=False, dtype="uint8")

    valid = np.load(mask_dir / "valid.npy")
    burn = np.load(exp / "derived_data" / "audit_scratch" / "burn_count.npy")

    np.save(mask_dir / "region.npy", region)
    profile = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                   crs=man["grid"]["crs"], transform=transform, compress="lzw")
    with rasterio.open(mask_dir / "region.tif", "w", **profile) as dst:
        dst.write(region, 1)

    stats = {}
    print(f"[regions] {len(names)} Regionen\n")
    print(f"    {'Region':30s} {'Pixel':>9s} {'gueltig':>9s} {'Feuertage':>10s} {'Anteil':>7s}")
    total_fire = int(burn[valid].sum())
    for n in names:
        c = code_of[n]
        m = region == c
        mv = m & valid
        fire = int(burn[mv].sum())
        stats[n] = {
            "code": c,
            "n_pixels": int(m.sum()),
            "n_valid_pixels": int(mv.sum()),
            "fire_pixel_days": fire,
            "fire_share": fire / max(1, total_fire),
        }
        print(f"    {n:30s} {int(m.sum()):9d} {int(mv.sum()):9d} {fire:10d} "
              f"{100*fire/max(1,total_fire):6.2f} %")

    unassigned = int((valid & (region == 0)).sum())
    print(f"\n[regions] gueltige Pixel ohne Region: {unassigned} "
          f"({100*unassigned/max(1,int(valid.sum())):.2f} %)")

    write_manifest(
        cfg,
        "d2_regions",
        args.config,
        {
            "provinces_geojson": args.provinces_geojson,
            "sha256_16": hashlib.sha256(Path(args.provinces_geojson).read_bytes()).hexdigest()[:16],
            "n_regions": len(names),
            "code_of": code_of,
            "regions": stats,
            "unassigned_valid_pixels": unassigned,
            "outputs": [str(mask_dir / "region.npy"), str(mask_dir / "region.tif")],
        },
    )


if __name__ == "__main__":
    main()
