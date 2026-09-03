# WildfirePrediction — Dense Spatiotemporal Survival Forecasting

Forschungsprototyp. Aktueller Stand: **Phase 1 (Repository- und Datenaudit) abgeschlossen.**
Es ist noch keine Trainingspipeline implementiert und kein Training gestartet.

## Pfade

| Rolle | Pfad |
|---|---|
| `REPOSITORY` | `/workspace/WildfirePrediction` (Host: `/home/nik/Workspace/WildfirePrediction`) |
| `CODE_DIR` | `wildfire/` |
| `RAW_DATA` | `/firedata` — **read-only**, wird niemals beschrieben |
| `EXPERIMENT_ROOT` | `/data/Wildfire` (Host: `/raid/data/nik/Wildfire`) |

Unter `EXPERIMENT_ROOT` liegen `configs/ data_manifests/ derived_data/ splits/ logs/
checkpoints/ predictions/ metrics/ figures/ artifacts/ reports/`. Grosse Daten,
Checkpoints und Ergebnisse gehoeren dorthin, nicht ins Git.

## Ausfuehrung

Alle Arbeitsschritte laufen ueber `sbatch`. Die Skripte liegen in `wildfire/slurm/`
und folgen den Konventionen der uebrigen Repos auf diesem Cluster
(pyxis/enroot-Container, Mounts `/home/nik/Workspace:/workspace` und
`/raid/data/nik:/data`). Der Rohdatenmount ist zusaetzlich als `:ro` gesetzt,
damit Read-only auf Mount-Ebene erzwungen wird.

```bash
sbatch wildfire/slurm/00_audit_env.sbatch        # Geo-Deps nach $EXPERIMENT_ROOT/pydeps
sbatch wildfire/slurm/01_audit_inventory.sbatch  # Stufe 0: Dateiinventar
sbatch wildfire/slurm/02_audit_tar_index.sbatch  # Stufe 1: tar-Header, Zeitachse
sbatch wildfire/slurm/03_audit_rasters.sbatch    # Stufe 2: CRS, Geometrie, Bandstatistik
sbatch wildfire/slurm/04_audit_fire_labels.sbatch# Stufe 3: Feuerlabels vollstaendig
sbatch wildfire/slurm/05_audit_caches.sbatch     # Stufe 4: HDF5-Caches, Splits
sbatch wildfire/slurm/06_audit_alignment.sbatch  # Stufe 5: Kanal-/Zeitrekonstruktion
sbatch wildfire/slurm/07_audit_label_def.sbatch  # Stufe 6: Labelregel, Shortcut-Test
sbatch wildfire/slurm/08_audit_report.sbatch     # Stufe 7: data_audit.md + Manifest
```

Jede Stufe schreibt genau ein JSON nach `$EXPERIMENT_ROOT/data_manifests/<stufe>.json`
mit Provenance-Kopf (Git-Commit, Config-Hash, Seed, Slurm-Job-ID, Host, Zeitpunkt).
Stufe 7 rendert Bericht und Manifest ausschliesslich aus diesen JSONs — es wird
keine Zahl von Hand uebertragen.

## Umgebung

Basis ist `/miniconda3/envs/vmtk_2` (numpy, pandas, h5py, tifffile, scipy,
matplotlib). `rasterio` und `pyproj` liegen persistent unter
`$EXPERIMENT_ROOT/pydeps` und werden ueber `PYTHONPATH` vorangestellt
(`wildfire/slurm/_common.sh`).

## Ergebnisse des Audits

- Bericht: `$EXPERIMENT_ROOT/reports/data_audit.md`
- Manifest: `$EXPERIMENT_ROOT/data_manifests/dataset_manifest.json`
