#!/bin/bash
#SBATCH --job-name=wf_sts_heads
#SBATCH --array=0-11%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Survival-Koepfe im Vergleich: dasselbe STS-Backbone (d 128, Tiefe 6), dasselbe
# Wuerfel-Vortraining und dieselbe Auswahl wie die DCTM-Checkpoints
# sts_cache_G_next_cube_d128_s*, nur der Kopf wechselt (models/surv_heads.py).
#   Index i: Kopf HEADS[i / 3], Seed i % 3
# Checkpoints: sts_cache_G_next_cube_d128_<kopf>_s<seed>_s<seed>.pt
# Danach: 57_paper_surv_eval und 58_paper_metrics_ckpt mit --models sts_<kopf>:...
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
HEADS=(lh deephit cox dctm_cloglog)
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
head=${HEADS[$((i / 3))]}; seed=$((i % 3))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --d 128 --depth 6 --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt \
    --head $head --tag "_d128_${head}_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
