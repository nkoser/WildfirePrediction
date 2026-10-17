#!/bin/bash
#SBATCH --job-name=wf_paper_hnmraw
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
# ---------------------------------------------------------------------------
# FWI-HNM-Gewichte direkt aus den Roh-Tars (FWI, MCD14A1), alle Tage der
# Trainingsjahre; dazu der Anteil der Werte am uint8-Rand (0/255).
#   EXTRA="--day-step 1 --pixel-frac 0.05"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.hnm_raw --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
