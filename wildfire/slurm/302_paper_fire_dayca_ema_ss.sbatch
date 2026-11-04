#!/bin/bash
#SBATCH --job-name=wf_dayca2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# fire_dayca nacheinander auf einer GPU: (1) EMA 0.995, 25 Epochen; (2) EMA + Scheduled Sampling (bis 7 eigene Tage).
# Ausgaben NEU: metrics/fire_dayca_ema/, metrics/fire_dayca_emass/.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
"$PY" -m wildfire.paper.fire_dayca --config "$CONFIG" --tag _ema --epochs 25 --ema 0.995 || rc=1
"$PY" -m wildfire.paper.fire_dayca --config "$CONFIG" --tag _emass --epochs 25 --ema 0.995 --ss 1.0 --ss-k 7 || rc=1
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
