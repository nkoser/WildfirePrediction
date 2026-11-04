#!/bin/bash
#SBATCH --job-name=wf_bb_abl
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Ablation des Rumpfs mit Logistic-Hazard-Kopf, sonst wie 60 (Wuerfel-Vortraining,
# 10 Epochen x 3000 Schritte, Auswahl nach Val-PR-AUC). Referenz: ..._d128_lh_s*.
#   0-2  volle Raum-Zeit-Attention          -> ..._d128_lh_joint_s<seed>
#   3-5  Zeit-Attention ueber 3 x 3-Nachbarn -> ..._d128_lh_local_s<seed>
#   6-8  3D-Stem (3 x 3 Pixel x 3 Tage)      -> ..._d128_lh_stem3d_s<seed>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
NAMES=(joint local stem3d); ARGS=("--attn joint" "--attn local" "--stem 3d")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --d 128 --depth 6 --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt \
    --head lh ${ARGS[$v]} --tag "_d128_lh_${NAMES[$v]}_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
