"""Stufe 1: Index aller tar-Archive - ausschliesslich ueber die Header.

Die Archive sind unkomprimiert (ustar), daher springt ``tarfile`` von
Header zu Header, ohne Nutzdaten zu lesen. Fuer ein 13,8-GB-Archiv mit
366 Mitgliedern sind das 366 Seeks. Es wird nichts entpackt und nichts
geschrieben.

Ergebnis je Modalitaet: Zeitachse, Kadenz, fehlende Tage, Duplikate,
Dateigroessenverteilung und die Namenskonvention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from wildfire.audit.common import (
    assert_readonly,
    human_bytes,
    load_config,
    write_manifest,
)

DAILY_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})\.tif{1,2}$", re.I)
ANNUAL_RE = re.compile(r"(?P<name>.+?)[_-](?P<y>\d{4})\.tif{1,2}$", re.I)


def index_tar(path_str: str) -> dict:
    path = Path(path_str)
    members: list[dict] = []
    error = None
    try:
        with tarfile.open(path, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                members.append({"name": m.name, "size": m.size})
    except Exception as exc:  # defekte Archive sollen den Lauf nicht killen
        error = f"{type(exc).__name__}: {exc}"

    dates, annual, other = [], [], []
    for m in members:
        base = Path(m["name"]).name
        if (hit := DAILY_RE.search(base)):
            dates.append(f"{hit['y']}-{hit['m']}-{hit['d']}")
        elif (hit := ANNUAL_RE.search(base)):
            annual.append(hit["y"])
        else:
            other.append(base)

    return {
        "tar": str(path),
        "tar_size_bytes": path.stat().st_size,
        "n_members": len(members),
        "error": error,
        "dates": sorted(dates),
        "annual_years": sorted(set(annual)),
        "unparsed_names": other[:20],
        "n_unparsed": len(other),
        "member_size_min": min((m["size"] for m in members), default=0),
        "member_size_max": max((m["size"] for m in members), default=0),
        "member_size_sum": sum(m["size"] for m in members),
        "example_members": [m["name"] for m in members[:5]],
        "subdirs": sorted({str(Path(m["name"]).parent) for m in members})[:10],
    }


def summarise_modality(name: str, per_tar: list[dict]) -> dict:
    all_dates: list[str] = []
    for t in per_tar:
        all_dates.extend(t["dates"])
    dup = {d: c for d, c in Counter(all_dates).items() if c > 1}
    uniq = sorted(set(all_dates))

    gaps: list[dict] = []
    coverage_by_year: dict[str, int] = defaultdict(int)
    if uniq:
        parsed = [dt.date.fromisoformat(d) for d in uniq]
        for d in parsed:
            coverage_by_year[str(d.year)] += 1
        for a, b in zip(parsed, parsed[1:]):
            delta = (b - a).days
            if delta > 1:
                gaps.append(
                    {"after": a.isoformat(), "before": b.isoformat(), "missing_days": delta - 1}
                )

    years = sorted({str(y) for t in per_tar for y in t["annual_years"]})
    n_members = sum(t["n_members"] for t in per_tar)
    cadence = "daily" if uniq else ("annual/static" if years else "unbekannt")

    # Erwartete Tage bei taeglicher Kadenz, um Vollstaendigkeit zu pruefen.
    expected = {}
    if uniq:
        for y, n in sorted(coverage_by_year.items()):
            yi = int(y)
            days_in_year = (dt.date(yi + 1, 1, 1) - dt.date(yi, 1, 1)).days
            expected[y] = {"present": n, "days_in_year": days_in_year, "missing": days_in_year - n}

    return {
        "modality": name,
        "n_tars": len(per_tar),
        "tar_size_bytes": sum(t["tar_size_bytes"] for t in per_tar),
        "tar_size_human": human_bytes(sum(t["tar_size_bytes"] for t in per_tar)),
        "n_members_total": n_members,
        "cadence": cadence,
        "date_min": uniq[0] if uniq else None,
        "date_max": uniq[-1] if uniq else None,
        "n_unique_dates": len(uniq),
        "n_duplicate_dates": len(dup),
        "duplicate_examples": dict(list(dup.items())[:10]),
        "annual_years": years,
        "n_gaps": len(gaps),
        "gaps": gaps[:50],
        "coverage_by_year": expected,
        "member_size_min": min((t["member_size_min"] for t in per_tar), default=0),
        "member_size_max": max((t["member_size_max"] for t in per_tar), default=0),
        "member_size_sum": sum(t["member_size_sum"] for t in per_tar),
        "uncompressed_vs_tar_ratio": (
            sum(t["member_size_sum"] for t in per_tar)
            / max(1, sum(t["tar_size_bytes"] for t in per_tar))
        ),
        "example_members": per_tar[0]["example_members"] if per_tar else [],
        "subdirs": per_tar[0]["subdirs"] if per_tar else [],
        "errors": [t["error"] for t in per_tar if t["error"]],
        "n_unparsed": sum(t["n_unparsed"] for t in per_tar),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    maps_root = Path(cfg["dataset_root"]) / "Entire_Canada_Maps"
    jobs: list[tuple[str, str]] = []
    for mod in cfg["modalities"]:
        for tar in sorted((maps_root / mod).glob("*.tar")):
            jobs.append((mod, str(tar)))
    print(f"[s1] {len(jobs)} tar-Archive, {args.workers} Prozesse")

    results: dict[str, list[dict]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for (mod, tar), res in zip(jobs, ex.map(index_tar, [j[1] for j in jobs], chunksize=1)):
            results[mod].append(res)
            flag = " FEHLER" if res["error"] else ""
            print(f"    {mod:10s} {Path(tar).name:24s} {res['n_members']:5d} Dateien{flag}")

    summary = {m: summarise_modality(m, sorted(results[m], key=lambda r: r["tar"])) for m in cfg["modalities"]}
    for m, s in summary.items():
        print(
            f"[s1] {m:10s} {s['cadence']:12s} {s['n_unique_dates']:5d} Tage "
            f"{str(s['date_min']):>10s}..{str(s['date_max']):>10s} "
            f"Luecken={s['n_gaps']:3d} Duplikate={s['n_duplicate_dates']:3d} "
            f"Members={s['n_members_total']}"
        )

    write_manifest(
        cfg,
        "s1_tar_index",
        args.config,
        {
            "maps_root": str(maps_root),
            "n_tars": len(jobs),
            "modalities": summary,
            "per_tar": {m: results[m] for m in cfg["modalities"]},
        },
    )


if __name__ == "__main__":
    main()
