#!/bin/bash
#SBATCH --job-name=wf_contagion2
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS Version 2: alle 30 Fenstertage, Brenndauer als gelernte Kurve
# ueber das Feueralter, 7 x 7-Kern. Gleiches Vortraining wie 60/63.
#   0-2  von Grund auf, Seeds 0-2          -> ..._d128_cg2_s<seed>
#   3-5  Rumpf aus Logistic Hazard (60)    -> ..._d128_cg2w_s<seed>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); CK=/data/Wildfire/checkpoints/paper
if [ $i -lt 3 ]; then tag=cg2; init=(); else
  tag=cg2w; init=(--init-from "$CK/sts_cache_G_next_cube_d128_lh_s{seed}_s{seed}.pt" --init-backbone-only); fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --d 128 --depth 6 --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt \
    --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve "${init[@]}" \
    --tag "_d128_${tag}_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
