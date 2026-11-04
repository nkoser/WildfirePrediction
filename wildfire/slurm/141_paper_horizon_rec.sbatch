#!/bin/bash
#SBATCH --job-name=wf_hz_rec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Paper-Protokoll an allen dt = 1..30 (paper/horizon_rec.py): 3D-U-Nets je trainiertem dt (1/7/15/30, 3 Seeds,
# Hyperparameter wie _next_paperhp_w0, diesmal mit Checkpoint) an jedem dt angewandt (native/nearest/interp)
# gegen F(dt + 1) aus STS-LH und Contagion-STS.
#   sbatch 141_...sbatch          voller Lauf -> metrics/paper_horizon/rec.{json,npz}, checkpoints/paper/horizon_rec/
#   sbatch 141_...sbatch smoke    Rauchtest (1 Seed, 2 Epochen) -> *_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${1:-}" = smoke ]; then
  a=(--seeds 0 --epochs 2 --tag _smoke)
else
  a=()
fi
"$PY" -m wildfire.paper.horizon_rec --config "$CONFIG" "${a[@]}" "${@:2}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
