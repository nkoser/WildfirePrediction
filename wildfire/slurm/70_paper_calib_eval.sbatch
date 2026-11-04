#!/bin/bash
#SBATCH --job-name=wf_calib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Kalibrierung nach Gruppen, PIT-Rekalibrierung und konformale untere Schranken
# (wildfire/paper/calib_eval.py) auf den surv_eval-Stichtagen. Ein Modell je
# Array-Index, damit die Modelle parallel auf eigenen GPUs laufen:
#   MODELS="art:glob art:glob ..." (Index i nimmt das i-te), TAG="_..."
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:-0}
read -ra M <<< "${MODELS:?MODELS fehlt}"
spec=${M[$i]:?kein Modell fuer Index $i}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.calib_eval --config "$CONFIG" --models "$spec" --tag "${TAG:-}_${spec%%:*}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
