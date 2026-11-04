#!/bin/bash
#SBATCH --job-name=wf_region_events
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Regionen-Auswertung, Schritt 1: Ereigniszeit je Pixel des 13x13-Ausschnitts fuer die
# zufaelligen Test-Stichtage (surv_eval_test, unveraendert), aus den MCD14A1-Rohdaten.
# Ausgabe NEU: derived_data/region_eval/pixel_events.npz (bricht ab, wenn sie existiert).
# Selbsttest gegen ev_patch / t_patch / ev_centre aus eval_meta.npz.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.region_eval --config "$CONFIG" --build-events
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
