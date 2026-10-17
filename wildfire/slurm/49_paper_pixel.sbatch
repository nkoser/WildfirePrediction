#!/bin/bash
#SBATCH --job-name=wf_paper_px
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Paper-Protokoll pro Pixel: Label je Pixel = Feuermaske dieses Pixels am
# Zieltag t; Persistenz, Gradient Boosting (eine Zeile je Pixel) und 3D-U-Net
# mit 13x13-Ausgabe, je dt. Braucht 40_paper_prepare (_L61).
#   CACHE=cache_G (Standard)   EXTRA="--methods hgb_px --seeds 0"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.pixel --config "$CONFIG" --cache "${CACHE:-cache_G}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
