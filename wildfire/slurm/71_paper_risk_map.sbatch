#!/bin/bash
#SBATCH --job-name=wf_riskmap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Dichte Pixel-Risikokarte F(tau) fuer ganz Kanada an einzelnen Stichtagen
# (wildfire/paper/risk_map.py), Seed-Ensemble, MODIS-Feuer danach als Referenz.
# Abbildungen danach mit 72_paper_risk_map_plot.sbatch (CPU).
#   PATTERN="<glob der Checkpoints>"  DATES="2023-06-01 2023-08-10"  TAG=cg2w_ema
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.risk_map --config "$CONFIG" ${PATTERN:+--pattern "$PATTERN"} \
    ${DATES:+--dates $DATES} ${TAG:+--tag "$TAG"} ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
