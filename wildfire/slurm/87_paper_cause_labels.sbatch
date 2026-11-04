#!/bin/bash
#SBATCH --job-name=wf_cause_labels
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Ursachen-Labels (Zuendung Blitz/Mensch/unbekannt, Ausbreitung, ohne Label) fuer die Ereignisse
# der Survival-Ziele, aus NFDB-Zuendungen und MODIS-Rohdaten. Nur gelesen: paper_pixel,
# surv_eval_test, region_eval, events, masks.
#   0  train  Wuerfel-Stichtage 2000-2019
#   1  eval   Test-Stichtage 2023-2025 (braucht 85_paper_region_events)
# Ausgabe NEU: derived_data/cause_labels/<teil>_cause.npz/.json (bricht ab, wenn sie existiert).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
part=$([ "$i" -eq 0 ] && echo train || echo eval)
"$PY" -m wildfire.paper.cause_labels --config "$CONFIG" --part $part ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
