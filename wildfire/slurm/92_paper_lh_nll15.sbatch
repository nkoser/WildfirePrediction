#!/bin/bash
#SBATCH --job-name=wf_lh_nll15
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Kontrolle zum gleichen Budget: Logistic Hazard d 64 wie 68 v5 (_d64_lh_ema_nll),
# aber bis 15 Epochen wie Contagion von Grund auf (83 v1, _d64_cg2_ema_nll). Die
# 10-Epochen-Referenz war bei allen drei Seeds in der letzten Epoche am besten.
#   -> ..._d64_lh_ema_nll15_s<seed>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
seed=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache cache_G --train-source cube \
    --target next_patch --epochs 15 --steps-per-epoch 3000 --seeds $seed --save-ckpt --ema 0.999 \
    --head lh --d 64 --depth 4 --select survnll --tag "_d64_lh_ema_nll15_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
