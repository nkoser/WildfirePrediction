#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_paper_pxlm
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Trainings-Stichtage fuer das ST-DCTM pro Pixel aus dem ganzen Wuerfel
# (2000-2019, Paper-Splits nach Jahren): Fenster 30 x 55 x 13 x 13 und je Pixel
# Zeit bis zum ersten MODIS-Feuer (bis 100 Tage). Danach 50 mit
#   EXTRA="--train-source cube --seeds 0 --tag _cube_s0"
#   EXTRA="--n-fire 120000 --n-random 120000"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.pixel_landmarks --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
