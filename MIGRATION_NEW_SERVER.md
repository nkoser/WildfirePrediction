# Umzug auf einen anderen Server

Stand: 2026-11-03, Branch `paper-reproduction` (mit uncommitteten Aenderungen).
Quelle: vcai `rtx-01-vcai`. Paket: `/scratch/shared/nik/transfer/` (gebaut mit
`wildfire/slurm/zz_pack_transfer.sbatch`).

## 1. Was der Zielserver braucht

| Anforderung | Warum |
|---|---|
| Slurm mit GPU-Partition | alle Skripte sind sbatch-Jobs |
| Apptainer (oder Singularity >= 3.8), `--nv` | jeder Job laeuft in `wildfire.sif` |
| NVIDIA-GPU, Treiber passend zum Image (nvcr-PyTorch-Basis) | Training/Eval |
| pro Job bis 140 GB RAM, 8 CPUs, 1 GPU (z. B. `99_paper_unet_centre`) | Speicherbedarf der Caches |
| Speicher: ~110 GB fuer das Paket; ~1,1 TB wenn alle derived_data mitkommen; +1,1 TB fuer Rohdaten nur falls der Cube neu gebaut wird | siehe Abschnitt 3 |

## 2. Paketinhalt

| Datei | Inhalt | Groesse ca. |
|---|---|---|
| `wildfire_core.tar` | `WildfirePrediction/` (Repo inkl. `.git`), `Wildfire/` (= EXPERIMENT_ROOT: Manifeste, Splits, Landmarks, Checkpoints, Metriken, Predictions, Logs, kleine derived_data), `containers/wildfire.sif`, `claude_memory/` | ~50 GB |
| `mpcfire_caches.tar` | `MPCFire/cache_A,G,H,Y` + `cache_G_{stratified,yearstrat,blockstrat,extreme_fire}` + README | ~56 GB |
| `*.sha256`, `*.list` | Pruefsummen, Inhaltslisten | |
| `derived_data_NOT_included.txt` | grosse Ordner, die fehlen | |

Nicht im Paket (Groessen auf vcai):

| `derived_data/...` | Groesse | Benoetigt fuer |
|---|---|---|
| `cube` | 341 GB | alles, was Features aus dem Cube zieht (STS, Landmarks neu, Kontagion) |
| `paper` | 94 GB | Paper-Protokoll-Rekonstruktion |
| `paper_pixel`, `_ys`, `_bs`, `_ea` | je ~91 GB | Pixel-Karten-Evaluation je Split |
| `surv_eval_test`, `_test_ea`, `_ys`, `_bs` | je 39 GB | Survival-Evaluation |
| `calib_cache_ext` | 43 GB | Kalibrierung |
| `paper_extra`, `paper_surv` | 22 / 21 GB | Zusatz-Samples, STS-DCTM |

Nur das mitnehmen, was die geplanten Jobs lesen (`--cache`, `--out`/Pfade im
jeweiligen sbatch-Skript und Modul pruefen). Alles andere ist mit den
vorhandenen sbatch-Skripten aus Rohdaten + Cube reproduzierbar, kostet aber Rechenzeit.

## 3. Rohdaten

Die Rohdaten sind der oeffentliche HF-Datensatz `AnonymousData4NeurIPS/MPCFire`
(~1,1 TB, davon 984 GB `Entire_Canada_Maps`). Nicht kopieren, sondern auf dem
Zielserver laden, falls der Cube neu gebaut werden muss:

```python
from huggingface_hub import snapshot_download
snapshot_download("AnonymousData4NeurIPS/MPCFire", repo_type="dataset", local_dir="<RAW>/MPCFire")
```

Fuer die Paper-Protokoll-Jobs auf `cache_G*` reicht `mpcfire_caches.tar`.

## 4. Uebertragen

```bash
# auf vcai (oder vom Zielserver aus ziehen)
rsync -avP /scratch/shared/nik/transfer/ ZIEL:/pfad/transfer/
# grosse derived_data gezielt, z. B. der Cube:
rsync -avP /scratch/shared/nik/Wildfire/derived_data/cube ZIEL:<EXP>/derived_data/
# auf dem Ziel pruefen
cd /pfad/transfer && sha256sum -c *.sha256
```

Fuer > 100 GB lohnt `rsync` direkt statt tar (fortsetzbar). Ohne direkte
Verbindung zwischen den Servern: ueber den eigenen Rechner oder eine Festplatte.

## 5. Auspacken und Pfade

Die Skripte sind auf diese Container-Pfade festgelegt; die Host-Pfade sind frei waehlbar:

| Container | vcai-Host | Override-Variable |
|---|---|---|
| `/workspace/WildfirePrediction` | `/home/nik/workspace/WildfirePrediction` | `WF_HOST_REPO` |
| `/data/Wildfire` | `/scratch/shared/nik/Wildfire` | `WF_HOST_EXP` |
| `/firedata` (ro, enthaelt `MPCFire/`) | `/cold/Niklas/FireData` | `WF_HOST_RAW` |
| Image | `/scratch/shared/nik/containers/wildfire.sif` | `WF_SIF` |

```bash
mkdir -p <ROOT> && cd <ROOT>
tar -xf wildfire_core.tar          # -> WildfirePrediction/ Wildfire/ containers/ claude_memory/
mkdir -p raw && tar -xf mpcfire_caches.tar -C raw      # -> raw/MPCFire/cache_*
export WF_HOST_REPO=<ROOT>/WildfirePrediction WF_HOST_EXP=<ROOT>/Wildfire \
       WF_HOST_RAW=<ROOT>/raw WF_SIF=<ROOT>/containers/wildfire.sif
```

Manifeste (z. B. `d4_cube.json`) speichern absolute **Container**-Pfade
(`/data/Wildfire/...`, `/firedata/...`) und bleiben gueltig, solange die Binds stimmen.

Zwei Stellen in jedem sbatch-Skript sind Host-Pfade und muessen angepasst werden:

1. `#SBATCH --output/--error=/scratch/shared/nik/Wildfire/logs/...` — beim Submit
   uebersteuern: `sbatch -o <ROOT>/Wildfire/logs/%x_%A_%a.out -e <ROOT>/Wildfire/logs/%x_%A_%a.err ...`
2. `source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh` (erste Zeile).

Am einfachsten auf dem Ziel dieselben Pfade per Symlink anlegen (dann aendert sich nichts):

```bash
ln -s <ROOT>/WildfirePrediction /home/$USER/workspace/WildfirePrediction
mkdir -p /scratch/shared/$USER && ln -s <ROOT>/Wildfire /scratch/shared/$USER/Wildfire
```

(funktioniert nur bei Benutzername `nik` bzw. wenn diese Verzeichnisse anlegbar sind;
sonst Pfade in Kopien der Skripte ersetzen). Ausserdem `#SBATCH --partition`/`--gres`
an den Zielcluster anpassen, falls dort nicht `gpu` / `gpu:1` gilt.

Claude-Memory: `claude_memory/` nach
`~/.claude/projects/<projektpfad-mit-bindestrichen>/memory/` kopieren.

## 6. Smoke-Test auf dem Ziel

```bash
mkdir -p <ROOT>/Wildfire/logs
sbatch -o ... -e ... wildfire/slurm/01b_probe_container.sbatch   # Container + GPU + Pakete
```

Danach einen kurzen Eval-Job auf vorhandenen Checkpoints laufen lassen und die
Metriken gegen `Wildfire/metrics/` von vcai vergleichen, bevor neue Trainings starten.

## 7. Jobs, die auf vcai beim Packen offen waren

Laufend: 5783 (wf_sts_c11), 5798 (wf_map_calib), 5879 (wf_sts_spread_tfmf), 5902 (wf_sts_spread_sf).
Pending/abhaengig: 5882–5884 (wf_firemap_bench).
Gehalten (JobHeldUser): 5784, 5812–5838 (Ablation ys/bs, sf_*), 5857–5866 (split_cgscr, t_scratch),
5897/5898 (split_region). Wenn ein Job auf dem neuen Server laufen soll: auf vcai
`scancel`, sonst rechnen beide doppelt.
