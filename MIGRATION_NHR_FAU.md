# Umzug auf NHR@FAU (Alex/Helma) und aktueller Projektstand

Stand: 2026-09-11, Git-Commit `708046d` auf `main`, Arbeitsbaum sauber.

Dieses Dokument hat zwei Teile:

1. **Projektstand** — was existiert, was gemessen wurde, was offen ist. Damit
   kann auf dem neuen Cluster ohne Rueckfragen weitergearbeitet werden.
2. **Umzugsplan** — wie Rohdaten, abgeleitete Daten und Code nach NHR@FAU
   kommen und was an den Slurm-Skripten geaendert werden muss.

---

## Teil 1: Projektstand

### 1.1 Datenquelle

Die Rohdaten sind der oeffentliche Hugging-Face-Datensatz
**`AnonymousData4NeurIPS/MPCFire`** (`repo_type="dataset"`, ca. 1,1 TB).
Sie wurden auf dem alten Cluster mit `snapshot_download` nach `/firedata/MPCFire`
geladen und dort read-only gemountet. Inhalt:

| Pfad | Groesse | Verwendung |
|---|---|---|
| `Entire_Canada_Maps/ERA5` | 334 GB | Cube (18 dyn. Kanaele) |
| `Entire_Canada_Maps/DEMs` | 130 GB | Cube (7 statische) |
| `Entire_Canada_Maps/FWI` | 112 GB | Cube (6 dyn.) |
| `Entire_Canada_Maps/MCD09CMG` | 75 GB | Cube (4 dyn., BT-Baender — **Leakage-Kandidat**) |
| `Entire_Canada_Maps/MCD09GA` | 75 GB | Cube (4 dyn.) |
| `Entire_Canada_Maps/MCD11A1` | 75 GB | Cube (4 dyn.) |
| `Entire_Canada_Maps/OSMs` | 75 GB | Cube (4 statische) |
| `Entire_Canada_Maps/MCD15A3H` | 38 GB | Cube (2 dyn.) |
| `Entire_Canada_Maps/MCD12Q1` | 19 GB | Cube (1 statisch) |
| `Entire_Canada_Maps/NDVI_EVI` | 19 GB | Cube (2 dyn.) |
| `Entire_Canada_Maps/VPD` | 19 GB | Cube (1 dyn.) |
| `Entire_Canada_Maps/Worldpop` | 19 GB | Cube (1 statisch) |
| `Entire_Canada_Maps/MCD14A1` | 81 MB | **Label** (MODIS Active Fire, binaer) |
| `cache_A/G/H/Y` | 4 x 14 GB | nur Audit; **im Projekt nicht verwendet** (Shortcut, s. Leakage-Bericht) |

Alle 13 Modalitaeten werden fuer den Cube gebraucht — ein Teil-Download per
`allow_patterns` spart praktisch nichts. Die vier Caches kann man weglassen,
wenn das Audit nicht wiederholt werden soll (−56 GB).

### 1.2 Was das Audit ergeben hat (Phasen 1–3 abgeschlossen)

Details in `$EXPERIMENT_ROOT/reports/data_audit.md` und `leakage_report.md`
(werden mitkopiert, siehe 2.4). Kernpunkte:

- Gitter **928 x 1968** Pixel, taeglich, **2000–2025 lueckenlos**. Aufloesung ist
  NICHT 1 km, sondern ca. 5 km N-S und 1,7–3,3 km O-W.
- Einzige Zielinformation: MCD14A1 Active-Fire-Maske. Keine echten
  Zuendpunkte, keine Perimeter, keine Ursachen.
- Zieldefinition daher **Proxy-Zuendung**: erster Feuertag nach Cooldown C = 7
  Tagen (265 075 Ereignisse; 35 710 in der Saison-/Patch-Definition der Splits).
- **Die mitgelieferten Caches sind unbrauchbar** (Feuermaske als Eingabekanal
  am Labeltag: 94,5 % Accuracy ohne Modell). Wir bauen alles selbst.
- Kanaele 38–41 (MCD09CMG Helligkeitstemperaturen) sind das Rohmaterial des
  Labels. Sie sind im Cube enthalten, aber muessen bei Modellvergleichen als
  Ablation (`no_bt`) mitgefuehrt werden.
- Offen: Gapfilling/Interpolation ueber den Zuendzeitpunkt (nicht pruefbar),
  Detektionsverzoegerung von MODIS (Cooldown daempft, beseitigt nicht).

### 1.3 Abgeleitete Daten unter `$EXPERIMENT_ROOT` (= `/data/Wildfire`)

| Pfad | Groesse | Reproduzierbar aus | Umzug |
|---|---|---|---|
| `derived_data/cube/` | **341 GB** | Rohdaten, `15_build_cube.sbatch` | neu bauen (oder rsync, wenn Bandbreite egal) |
| `derived_data/{masks,events,features,landmarks,baselines,audit_scratch}`, `norm_stats.npz` | ~570 MB | Pipeline 10–27 | **rsync** |
| `data_manifests/` (41 JSONs, Provenance jeder Stufe) | 7 MB | — | **rsync** |
| `splits/` (temporal, spatial_bc, spatial_east, extreme_2023) | 9 MB | `14_build_splits` | **rsync** |
| `artifacts/` (NFDB_point_txt.zip, Natural-Earth-GeoJSONs + sha256) | 30 MB | **nicht** aus HF; externe Downloads | **rsync** |
| `checkpoints/` (b2_unet_temporal{,_smoke}, b4_{bernstein,discrete}_best.pt) | 30 MB | Training | **rsync** |
| `reports/`, `figures/`, `logs/`, `experiment_registry.jsonl` | < 1 MB | — | **rsync** |
| `pydeps-cp311/` (rasterio, pyproj fuer Python 3.11) | — | `01c_setup_deps.sbatch` | neu bauen |

Cube-Layout: je Jahr `dyn_<jahr>.mmap` (T=183 Saisontage 01.04.–30.09., H, W, 42)
uint8 und `stat_<jahr>.npy` (H, W, 13). Manifest `data_manifests/d4_cube.json`.

### 1.4 Pipeline (alle Schritte via `sbatch`, Skripte in `wildfire/slurm/`)

```
00–09  Audit (s0–s8)                     -> data_manifests/s*.json, reports/data_audit.md
10  build_masks      -> d0_masks.json    Land-/Gueltigkeitsmaske, Gitter
11  build_events     -> d1_events.json   Proxy-Zuendungen, Ereignis-IDs
12  ignition_qc      -> d1b
13  build_regions    -> d2_regions.json  Provinzen/Oekozonen
14  build_splits     -> d3_splits.json   temporal / spatial_bc / spatial_east / extreme_2023
15  build_cube       -> d4_cube.json     ***341 GB, IO-gebunden, je Jahr ~8 GB***
15b build_cube_array               Slurm-Array-Variante (ein Jahr je Task)
16  verify_cube      -> d5
17  build_fire_index -> d6
18  build_landmarks  -> d7  (derived_data/landmarks/temporal_{train,val,test}.npz)
19  build_geometry   -> d8
20  nfdb_validation  -> d9  (NFDB-Abgleich, braucht artifacts/NFDB_point_txt.zip)
21  run_tests                          pytest wildfire/tests
22  sanity_checks    -> d10
23  baseline_b0      -> m0_climatology.json
24  leakage_report   -> reports/leakage_report.md
25  build_features   -> f1_features.json (derived_data/features/*.npz, 218 Merkmale)
26  baseline_b1      -> m1_tabular.json
27  norm_stats       -> d11_norm_stats.json, derived_data/norm_stats.npz
28  train_b2         -> m2_b2_unet_temporal.json   (GPU; SMOKE=1 fuer Rauchtest)
29  baseline_b4      -> m4_{discrete,bernstein}.json, m5_monotonicity.json
30  nfdb_events      -> d12
31  predict_map                        Risikokarte aus Checkpoint (CKPT=, DATE=)
```

Fuer den Neuaufbau auf dem Zielcluster muessen nur **10 -> 15 (bzw. 15b) -> 16**
laufen, alles andere wird per rsync uebernommen. Danach mit `21_run_tests`
und `22_sanity_checks` pruefen, dass Cube und Manifeste zusammenpassen
(`d5_verify_cube.json` vergleicht Pruefsummen gegen `d4_cube.json`).

### 1.5 Ergebnisse bisher (Split `temporal`: Train 2000–17, Val 2018–20, Test 2021–25)

ROC-AUC auf **Test** je Horizont (Tage), Proxy-Zuendung, Horizont 100:

| Modell | @1 | @7 | @14 | @30 | @60 | @100 | Bemerkung |
|---|---|---|---|---|---|---|---|
| B0 Klimatologie (pixel_bin) | — | — | — | 0.70 | — | — | m0_climatology.json |
| B1 LogReg (218 Merkmale, isotonisch) | 0.846 | 0.821 | 0.805 | 0.786 | 0.779 | 0.787 | m1_tabular.json |
| B1 HGB full | 0.846 | 0.837 | 0.824 | **0.807** | 0.802 | **0.813** | bestes Modell bisher |
| B1 HGB no_geo | 0.847 | 0.835 | 0.823 | 0.805 | 0.799 | 0.810 | Geo-Merkmale bringen ~nichts |
| B4 MLP discrete-hazard (100 Ausg.) | 0.865 | 0.842 | 0.823 | 0.805 | 0.798 | 0.807 | m4_discrete.json |
| B4 MLP Bernstein (Grad 12) | 0.790 | 0.806 | 0.805 | 0.799 | 0.800 | 0.809 | m4_bernstein.json |
| B2 U-Net 33x33, 5 Zeitscheiben | — | — | — | (val 0.78) | — | — | **noch keine Test-Auswertung** |

B2 (`checkpoints/b2_unet_temporal/best.pt`, 1,16 M Parameter, 228 Eingabekanaele):
Early Stopping nach 4 Epochen, bestes val-NLL 0.0697 in Epoche 0; val-NLL
danach instabil (0.070 -> 0.111 -> 0.084 -> 0.128) bei gleichzeitig steigendem
val-AUC@30 (0.766 -> 0.794). Lauf auf B200, ~26 min/Epoche bei 1,5 M Landmarks
je Epoche, 24 Dataloader-Prozesse, 400 GB RAM (Seitencache fuer den
Trainings-Cube von 253 GB).

Monotonie-Check (m5): CIF ist bei beiden B4-Koepfen monoton, Hazard faellt
ueber den Horizont (Faktor 6,6 bzw. 13) — plausibel, weil die letzte
Beobachtung an Informationswert verliert.

### 1.6 Offene Punkte / naechste Schritte (in dieser Reihenfolge)

1. **B2 sauber abschliessen**: Test-Auswertung fehlt; die val-NLL-Instabilitaet
   deutet auf Kalibrierungs-/LR-Problem hin (AUC steigt, NLL faellt nicht).
   Kandidaten: kleinere LR, `init_bias`, AMP erst nach FP32-Vergleich
   (Config-Kommentar), Verlustgewichtung wie in B4.
2. **`31_predict_map`** wurde noch nie ausgefuehrt (`predictions/` ist leer).
3. **B3** (raeumlich-zeitliche Baseline, Plan Phase 5) ist nicht implementiert.
4. **Spatial-OOD- und Extreme-Season-Splits** existieren als CSV, wurden aber
   fuer keine Baseline ausgewertet — bisher nur `temporal`.
5. **Ablation `no_bt`** (ohne MCD09CMG-Kanaele 38–41) fuer alle Baselines.
6. Hauptmodell (Phase 6, DCTM-Head) noch nicht begonnen.
7. `experiment_registry.jsonl` enthaelt ueberholte Zeilen mit altem
   `split_hash` (`db48c4…` vor Neuberechnung der Splits am 31.08.); gueltig ist
   `91e31be4dfa5e7d3`.

---

## Teil 2: Umzug nach NHR@FAU

Zielsystem: NHR@FAU, GPU-Cluster **Alex** (A100/A40) oder **Helma** (H100/H200).
Quellen: [Cluster-Uebersicht](https://doc.nhr.fau.de/clusters/overview/),
[Dateisysteme](https://doc.nhr.fau.de/data/filesystems/),
[Datentransfer](https://doc.nhr.fau.de/data/copying/),
[Alex](https://doc.nhr.fau.de/clusters/alex/),
[PyTorch](https://doc.nhr.fau.de/apps/pytorch/).

### 2.1 Was anders ist als auf dem alten Cluster

| | alt (dgx02) | NHR@FAU |
|---|---|---|
| Container | pyxis/enroot (`--container-image`, `--container-mounts`) | **Apptainer** (`apptainer exec --bind …`) |
| Internet auf Compute-Nodes | ja | **nur ueber Proxy** `http://proxy.nhr.fau.de:80` |
| GPU-Anforderung | `--partition=gpu --gres=gpu:1` | `--gres=gpu:a40:1` bzw. `--gres=gpu:a100:1 -C a100_80`, Partition = GPU-Typ |
| Walltime | 12 h+ | **max. 24 h** |
| Env-Weitergabe | Standard | `#SBATCH --export=NONE` + `unset SLURM_EXPORT_ENV` (FAU-Konvention) |
| RAM je GPU | frei waehlbar (2 TB Knoten) | fest: 60–240 GB Host-RAM je GPU → **400 GB Seitencache wie in `28_train_b2` gibt es nicht** |
| lokaler Scratch | — | `$TMPDIR` auf NVMe, 7 TB (a40) / 14 TB (a100) |
| Login | direkt | ueber Dialogserver `csnhr.nhr.fau.de` |

### 2.2 Speicherplatz auf NHR@FAU

| FS | Quota | Plan |
|---|---|---|
| `$HOME` | 100 GB, Backup | Repo, Apptainer-Image (.sif) |
| `$HPCVAULT` | 1 TB, Backup | kleine, nicht reproduzierbare Dinge (Manifeste, Splits, Artefakte, Checkpoints, Reports) |
| `$WORK` | Tier3: 1000 GB / NHR-Projekt: laut Antrag, **kein Backup** | Rohdaten 1,1 TB + Cube 340 GB — **nur wenn Projektquota ≥ 1,6 TB** |
| `$FASTTMP` (Lustre) | Groesse unbegrenzt, Dateianzahl limitiert, High-Watermark-Loeschung | Alternative fuer Rohdaten (338 grosse tars) und Cube (52 grosse Dateien); beides reproduzierbar, Verlust also verschmerzbar. Vorher pruefen, ob auf Alex gemountet. |
| `$TMPDIR` | NVMe, Job-Laufzeit | Cube-Jahre am Jobanfang hierher kopieren (siehe 2.6) |

**Erste Aktion auf dem Cluster:**

```bash
shownicerquota.pl            # Quota $HOME/$WORK/$HPCVAULT
ls -d /lustre 2>/dev/null    # gibt es $FASTTMP auf diesem Cluster?
echo $WORK $HPCVAULT $FASTTMP
```

Zielstruktur (Vorschlag; `$WORK` ggf. durch `$FASTTMP` ersetzen):

```
$HOME/WildfirePrediction          Repo (git clone)
$HOME/containers/wildfire.sif     Apptainer-Image
$WORK/MPCFire                     Rohdaten von HF          (-> /firedata/MPCFire im Container)
$WORK/Wildfire                    EXPERIMENT_ROOT          (-> /data/Wildfire im Container)
$HPCVAULT/Wildfire_backup         rsync-Kopie der kleinen Verzeichnisse
```

### 2.3 Rohdaten direkt von Hugging Face laden

Kein Transfer ueber den alten Server noetig. Der Download laeuft als Slurm-Job
mit Proxy; er ist resumable und ueberspringt vorhandene Dateien. Bei 1,1 TB
sind vermutlich 2–3 Jobs a 24 h noetig — einfach erneut submitten.

```bash
#!/bin/bash
#SBATCH --job-name=mpcfire_dl
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1          # Alex vergibt CPUs nur mit GPU
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME=$WORK/hf_cache      # nicht $HOME (100 GB)
# HF_TOKEN vorher einmalig per `huggingface-cli login` in $HF_HOME ablegen,
# nicht ins Skript schreiben.

module add python
python -m pip install --user -q huggingface_hub
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="AnonymousData4NeurIPS/MPCFire", repo_type="dataset",
    local_dir=os.path.join(os.environ["WORK"], "MPCFire"),
    max_workers=8,
    # ignore_patterns=["cache_*/*"],   # -56 GB, wenn kein Audit-Rerun geplant
)
print("fertig")
EOF
```

`hf_transfer` hinter dem Proxy erst mal nicht aktivieren (Proxy-Unterstuetzung
dort unzuverlaessig). Nach dem Download Vollstaendigkeit pruefen:
`find $WORK/MPCFire/Entire_Canada_Maps -name '*.tar' | wc -l` muss **338** ergeben
(vgl. `data_manifests/s1_tar_index.json`).

### 2.4 Kleine Verzeichnisse vom alten Cluster kopieren (< 1 GB)

Vom **alten** Cluster aus (Host-Pfade, nicht Container-Pfade). `-e` tunnelt
ueber den Dialogserver; `<user>`/`<gruppe>` einsetzen.

```bash
SRC=/raid/data/nik/Wildfire
DST='<user>@alex.nhr.fau.de:/home/woody/<gruppe>/<user>/Wildfire'   # = $WORK/Wildfire
rsync -avP -e 'ssh -J csnhr.nhr.fau.de' \
  --include='derived_data/' \
  --include='derived_data/masks/***' --include='derived_data/events/***' \
  --include='derived_data/features/***' --include='derived_data/landmarks/***' \
  --include='derived_data/baselines/***' --include='derived_data/audit_scratch/***' \
  --include='derived_data/norm_stats.npz' \
  --exclude='derived_data/*' \
  --exclude='pydeps-*' --exclude='logs/*.err' \
  "$SRC/" "$DST/"
```

Das ueberspringt `derived_data/cube` (341 GB) und `pydeps-*`. Anschliessend
dieselbe Kopie zusaetzlich nach `$HPCVAULT/Wildfire_backup` (Backup).

Falls die Bandbreite es hergibt, kann der Cube stattdessen auch kopiert werden
(`--include='derived_data/cube/***'`) — spart den Neuaufbau (Schritte 10/15/16),
aber 341 GB ueber SSH dauern typischerweise laenger als der Neuaufbau.

### 2.5 Apptainer-Image statt enroot

Das alte Image `/raid/data/nik/vesselgpt_5` (Env `/miniconda3/envs/vmtk_2`,
Python 3.11.8, torch 2.10+cu128, numpy, scipy, h5py, sklearn, matplotlib,
pytest; rasterio/pyproj aus `$DEPS`) muss als `.sif` bereitgestellt werden:

```bash
# auf dem alten Cluster, falls das Image als Docker-Archiv vorliegt:
apptainer build wildfire.sif docker-archive://vesselgpt_5.tar
# oder aus einer Registry / einem Dockerfile neu bauen und dann:
scp -J csnhr.nhr.fau.de wildfire.sif <user>@alex.nhr.fau.de:containers/
```

Alternativ ein frisches Image von NGC (`nvcr.io/nvidia/pytorch:25.xx-py3`) mit
`pip install h5py rasterio pyproj scikit-learn pytest` — FAU raet ausdruecklich
von grossen conda/pip-Envs auf den Shared-Filesystemen ab (viele kleine
Dateien), ein einzelnes `.sif` ist die empfohlene Form.

**Bind-Mounts so setzen, dass die Container-Pfade unveraendert bleiben** — dann
muss am Python-Code und an `configs/audit.yaml` nichts geaendert werden:

```bash
apptainer exec --nv \
  --bind $HOME/WildfirePrediction:/workspace/WildfirePrediction \
  --bind $WORK/MPCFire:/firedata/MPCFire:ro \
  --bind $WORK/Wildfire:/data/Wildfire \
  --bind $TMPDIR:/tmp \
  $HOME/containers/wildfire.sif  <befehl>
```

### 2.6 Aenderungen an den Slurm-Skripten

Betroffen ist ausschliesslich `wildfire/slurm/` (der Python-Code kennt nur
`/firedata`, `/data/Wildfire`, `/workspace/WildfirePrediction`).

1. **Header** in allen `*.sbatch`:
   - `--container-image`, `--container-mounts`, `--container-remap-root`,
     `--container-writable` entfernen.
   - `--partition=gpu --gres=gpu:1` → `--gres=gpu:a40:1` (oder `a100`). CPU-Jobs
     (Audit, Cube-Bau) ebenfalls mit einer GPU anfordern — Alex hat keine reine
     CPU-Partition; alternativ diese Schritte auf **Fritz/Woody** (CPU-Cluster,
     gleiches `$WORK`) laufen lassen.
   - `--output`/`--error`: `/raid/data/nik/Wildfire/logs/…` → `$WORK/Wildfire/logs/…`
     (Slurm expandiert `$WORK` im Header nicht; absoluten Pfad eintragen).
   - `--time` ≤ 24:00:00; `--mem` entfaellt (fest je GPU).
   - `#SBATCH --export=NONE` ergaenzen, danach `unset SLURM_EXPORT_ENV`.
2. **`_common.sh`**: Der Kommentarblock ueber Host- vs. Container-Logpfade wird
   hinfaellig. `REPO`/`EXP` bleiben, weil die Binds die Pfade nachbilden.
   Die Interpreter-Suche (`/miniconda3/envs/vmtk_2/bin/python`) passt zum alten
   Image; bei einem neuen Image `PY=` entsprechend setzen oder die
   Kandidatenliste ergaenzen. Proxy-Exports fuer Jobs mit Netzzugriff hier
   zentral setzen.
3. **Aufruf**: statt `"$PY" -m wildfire.xyz …` innerhalb des Containers wird
   das ganze Skript mit `apptainer exec … bash -c '…'` gewrappt, oder man
   sourced `_common.sh` innerhalb von `apptainer exec`. Sauberste Variante: ein
   `wildfire/slurm/_apptainer.sh` mit der Bind-Liste, das jedes sbatch-Skript
   fuer seinen Python-Aufruf benutzt.
4. **`28_train_b2.sbatch` / I/O**: Das Training liest die Cube-mmaps
   random-access. Auf dem alten Knoten lief das ueber 400 GB Seitencache; auf
   Alex gibt es je GPU nur 60–240 GB. Deshalb am Jobanfang die benoetigten
   Jahre nach `$TMPDIR` (NVMe) kopieren und den Cube-Pfad umbiegen:
   ```bash
   mkdir -p $TMPDIR/cube
   for y in $(seq 2000 2020); do cp $WORK/Wildfire/derived_data/cube/{dyn,stat}_$y.* $TMPDIR/cube/; done
   # dann --bind $TMPDIR/cube:/data/Wildfire/derived_data/cube
   ```
   Trainingsjahre 2000–2017 + Val 2018–2020 sind 21 x ~13 GB ≈ 280 GB, das
   Kopieren kostet einige Minuten pro Job. `workers: 24` in `b2_unet.yaml` zu
   den je GPU verfuegbaren 16–32 Kernen passen.
5. **Checkpoint-Resume** wegen 24-h-Limit: `wildfire/train.py` schreibt zwar
   `last.pt` (Zeile ~180), hat aber **keinen `--resume`-Pfad** — ein
   abgebrochener Lauf faengt heute von vorn an. Bei 8 Epochen a ~26 min ist
   das fuer B2 noch unkritisch (< 4 h), fuer das Hauptmodell muss Resume
   (Epoche, Optimizer, Scheduler, RNG-Zustand) nachgeruestet werden.
6. **`15b_build_cube_array.sbatch`** ist fuer den Neuaufbau das richtige
   Skript (ein Jahr je Array-Task, parallel). Bei `$FASTTMP` als Ziel die
   Dateianzahl im Blick behalten (52 Dateien, unkritisch).

### 2.7 Reihenfolge auf dem neuen Cluster

```
1.  Zugang einrichten, ssh -J csnhr.nhr.fau.de alex.nhr.fau.de
2.  shownicerquota.pl; Entscheidung $WORK vs. $FASTTMP fuer Rohdaten + Cube
3.  git clone <repo> $HOME/WildfirePrediction
4.  .sif bereitstellen ($HOME/containers/wildfire.sif)
5.  sbatch Download-Job (2.3); parallel dazu:
6.  rsync kleine Verzeichnisse (2.4) nach $WORK/Wildfire und $HPCVAULT
7.  Slurm-Skripte anpassen (2.6), 01c_setup_deps (rasterio/pyproj) laufen lassen
8.  sbatch 21_run_tests                  -> pytest gruen im neuen Image
9.  sbatch 10_build_masks                -> d0_masks.json muss zum kopierten identisch sein
10. sbatch 15b_build_cube_array (26 Tasks), dann 16_verify_cube
    -> d5_verify_cube.json muss gegen kopiertes d4_cube.json passen
11. SMOKE=1 sbatch 28_train_b2           -> Rauchtest auf A100/A40
12. Weiter mit Punkt 1.6 (B2 abschliessen, Test-Auswertung, predict_map)
```

Jeder Lauf schreibt Git-Commit, Slurm-Job-ID, Host und Config-Hash in sein
Manifest und in `experiment_registry.jsonl`; der Clusterwechsel ist damit in
der Provenance sichtbar (`hostname`, `gpu`).
