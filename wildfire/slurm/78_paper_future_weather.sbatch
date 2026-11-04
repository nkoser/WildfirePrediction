#!/bin/bash
#SBATCH --job-name=wf_future_weather
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Wetter-Oracle, Daten (future_weather): beobachtetes Wetter (Kanaele 1..25) der Tage L+1..L+30 je
# Wuerfel-Stichtag -> derived_data/paper_pixel/train_Xf.npy, train_kf.npy, Manifest p26_future_weather.
# Danach 74 v31 (--future-days 30).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.future_weather --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
