#!/bin/bash
#SBATCH --job-name=wf_ea_heads
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Kopfvergleich (Folie 14 von Survival_Models_Wildfire) auf flaechentreuen Ausschnitten (80):
# gleiche Laeufe wie 56 (DCTM), 60 (Koepfe) und 65 (Contagion v2, warm aus LH), nur
# --cache cache_G_ea --cube-dir paper_pixel_ea --surv-dir surv_eval_test_ea.
#   Index i: Variante i / 3, Seed i % 3
#   0 dctm   1 lh   2 deephit   3 cox   4 dctm_cloglog   5 cg2w (braucht 1: Indizes 3-5)
# Checkpoints: sts_cache_G_ea_next_cube_d128[_<kopf>]_s<seed>_s<seed>.pt
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); CK=/data/Wildfire/checkpoints/paper
case $v in
  0) args=(--tag "_d128_s${seed}") ;;
  1) args=(--head lh --tag "_d128_lh_s${seed}") ;;
  2) args=(--head deephit --tag "_d128_deephit_s${seed}") ;;
  3) args=(--head cox --tag "_d128_cox_s${seed}") ;;
  4) args=(--head dctm_cloglog --tag "_d128_dctm_cloglog_s${seed}") ;;
  5) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --init-backbone-only
           --init-from "$CK/sts_cache_G_ea_next_cube_d128_lh_s{seed}_s{seed}.pt" --tag "_d128_cg2w_s${seed}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache cache_G_ea --train-source cube \
    --cube-dir paper_pixel_ea --surv-dir surv_eval_test_ea \
    --target next_patch --d 128 --depth 6 --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt \
    "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
