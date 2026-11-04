"""CWFIS Reported Fires (taegliche Lageberichte der Feuerbehoerden) per WFS nach artifacts/cwfis_reportedfires/.

Die Download-Knoepfe der CWFIS-Seite bieten nur "Current Year" und "Last Year"; der WFS-Layer
public:cwfif_national_reportedfires haelt aber Statusreihen ab 2010. Je Status-Jahr wird seitenweise geholt
(Server-Limit 10 000 Zeilen, sortiert nach id) und als status_<jahr>.csv.gz gespeichert. Eine Zeile = ein
Statusintervall [record_start, record_end) eines Brandes mit stage_of_control_status (OC/BH/UC/EX), response_type
(FUL/MOD/MON), fire_size, Koordinate (latitude/longitude, Grad).

Vorhandene Jahresdateien werden nicht ueberschrieben (fuer einen Neuabzug ein neues --out waehlen).
Ausgabe NEU: <out>/status_<jahr>.csv.gz, <out>/fetch_log.json (Zeilen je Jahr, URL, Zeitpunkt).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

WFS = "https://geoserver.cwfif.nrcan.gc.ca/geoserver/ows"
LAYER = "public:cwfif_national_reportedfires"
PAGE = 10000


def page(year: int, start: int) -> list[list[str]]:
    q = {"service": "WFS", "version": "2.0.1", "request": "GetFeature", "typeName": LAYER, "outputFormat": "csv",
         "sortBy": "id A", "count": str(PAGE), "startIndex": str(start), "CQL_FILTER": f"status_year={year}"}
    url = WFS + "?" + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return list(csv.reader(io.TextIOWrapper(r, encoding="utf-8")))
        except Exception as e:  # noqa: BLE001 - Netz: erneut versuchen
            print(f"[fetch] {year} @{start}: {e} (Versuch {attempt + 1})", flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"ABBRUCH: {year} @{start} nicht ladbar")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/Wildfire/artifacts/cwfis_reportedfires")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2010, 2027)))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {"wfs": WFS, "layer": LAYER, "fetched_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
           "job": os.environ.get("SLURM_JOB_ID"), "years": {}}
    for y in args.years:
        dst = out / f"status_{y}.csv.gz"
        if dst.exists():
            print(f"[fetch] {dst.name} vorhanden, uebersprungen", flush=True)
            continue
        header, rows, start = None, [], 0
        while True:
            p = page(y, start)
            if not p:
                break
            header = header or p[0]
            if p[0] != header:
                raise SystemExit(f"ABBRUCH: Kopfzeile aendert sich in {y} @{start}")
            rows += p[1:]
            if len(p) - 1 < PAGE:
                break
            start += PAGE
            time.sleep(1)
        ids = {r[header.index("id")] for r in rows} if header else set()
        tmp = dst.with_suffix(".part")
        with gzip.open(tmp, "wt", newline="") as fh:
            w = csv.writer(fh)
            if header:
                w.writerow(header)
            w.writerows(rows)
        os.replace(tmp, dst)
        log["years"][y] = {"rows": len(rows), "unique_ids": len(ids)}
        print(f"[fetch] {y}: {len(rows):,} Zeilen ({len(ids):,} ids)", flush=True)
    lp = out / "fetch_log.json"
    old = json.loads(lp.read_text()) if lp.exists() else {"runs": []}
    old["runs"].append(log)
    lp.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
