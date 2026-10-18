#!/bin/bash
#SBATCH --job-name=wf_paper_pm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Alle Paper-Metriken (P, R, F1, PR-AUC, ROC-AUC, ECE, MCE, Brier) im Protokoll
# des Papers: STS-DCTM, ST-DCTM pro Pixel (Platt auf dem Paper-Val-Split) und
# der Nachbau des Paper-Modells. Braucht 40_paper_prepare (_L61), 54 fuer D+1.
#   EXTRA="--target cache"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache "${CACHE:-cache_G}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
