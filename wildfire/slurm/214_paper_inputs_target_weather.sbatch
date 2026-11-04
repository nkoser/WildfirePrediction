#!/bin/bash
#SBATCH --job-name=wf_inputs_tgt_w
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Ziel D gegen D+1 fuer alle Eingabesaetze inkl. weather (sts_inputs_target_weather.py), nach 212 und 213.
# Ausgaben NEU: p46_sts_inputs_target${TAG}, metrics/sts_inputs_target${TAG}/summary.md (TAG Standard _weather).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_inputs_target_weather --config "$CONFIG" --tag "${TAG:-_weather}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
