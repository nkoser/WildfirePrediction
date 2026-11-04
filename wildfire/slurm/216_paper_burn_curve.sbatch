#!/bin/bash
#SBATCH --job-name=wf_burn_curve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Gelernte Brenndauer (cg_ref aus 166, cg_episode aus 215) gegen Feuer-Episoden in den Daten
# (wildfire/paper/contagion_burn_curve.py). Ausgaben NEU: metrics/contagion_burn_curve${TAG}/, p48_contagion_burn_curve${TAG}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_burn_curve --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
