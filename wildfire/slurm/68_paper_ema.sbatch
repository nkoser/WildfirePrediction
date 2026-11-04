#!/bin/bash
#SBATCH --job-name=wf_ema
#SBATCH --array=0-38
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# EMA der Gewichte (--ema 0.999), sonst wie die Referenzlaeufe (Wuerfel-Vortraining,
# 10 Epochen x 3000 Schritte, Auswahl nach Val-PR-AUC mit den EMA-Gewichten):
#   0-2  Logistic Hazard, d 128, 6 Bloecke          -> ..._d128_lh_ema_s<seed>    (Referenz 60: _d128_lh)
#   3-5  Contagion v2, Rumpf aus Logistic Hazard     -> ..._d128_cg2w_ema_s<seed>  (Referenz 65: _d128_cg2w)
#   6-8  Logistic Hazard, kleiner: d 64, 4 Bloecke   -> ..._d64_lh_ema_s<seed>
#   9-11 Contagion v2 klein, Rumpf aus 6-8          -> ..._d64_cg2w_ema_s<seed>
#  12-14 Contagion v2 klein, von Grund auf          -> ..._d64_cg2_ema_s<seed>
#  15-17 wie 6-8, Auswahl nach Survival-NLL (2020-22) -> ..._d64_lh_ema_nll_s<seed>
#  18-20 wie 6-8, Dropout 0,2                       -> ..._d64_lh_ema_p2_s<seed>
#  21-23 LH d 128, Auswahl nach Survival-NLL, bis 15 Epochen      -> ..._d128_lh_ema_nll_s<seed>
#  24-26 Contagion d 64, Rumpf aus 15-17, Auswahl nach NLL, 15 Ep. -> ..._d64_cg2w_ema_nll_s<seed>
#  27-29 Contagion d 128, Rumpf aus 21-23, Auswahl nach NLL, 15 Ep. -> ..._d128_cg2w_ema_nll_s<seed>
#  30-32 LH d 64, Auswahl nach NLL, X-CAL lambda 10   -> ..._d64_lh_ema_nll_xcal10_s<seed>
#  33-35 LH d 64, Auswahl nach NLL, X-CAL lambda 100  -> ..._d64_lh_ema_nll_xcal100_s<seed>
#  36-38 LH d 64, Auswahl nach NLL, Joint-Attention  -> ..._d64_lh_ema_nll_joint_s<seed>   (Referenz 15-17)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); CK=/data/Wildfire/checkpoints/paper
case $v in
  0) args=(--head lh --d 128 --depth 6 --tag "_d128_lh_ema_s${seed}") ;;
  1) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --d 128 --depth 6
           --init-from "$CK/sts_cache_G_next_cube_d128_lh_s{seed}_s{seed}.pt" --init-backbone-only
           --tag "_d128_cg2w_ema_s${seed}") ;;
  2) args=(--head lh --d 64 --depth 4 --tag "_d64_lh_ema_s${seed}") ;;
  3) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --d 64 --depth 4
           --init-from "$CK/sts_cache_G_next_cube_d64_lh_ema_s{seed}_s{seed}.pt" --init-backbone-only
           --tag "_d64_cg2w_ema_s${seed}") ;;
  4) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --d 64 --depth 4
           --tag "_d64_cg2_ema_s${seed}") ;;
  5) args=(--head lh --d 64 --depth 4 --select survnll --tag "_d64_lh_ema_nll_s${seed}") ;;
  6) args=(--head lh --d 64 --depth 4 --dropout 0.2 --tag "_d64_lh_ema_p2_s${seed}") ;;
  7) args=(--head lh --d 128 --depth 6 --select survnll --epochs 15 --tag "_d128_lh_ema_nll_s${seed}") ;;
  8) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --d 64 --depth 4 --select survnll --epochs 15
           --init-from "$CK/sts_cache_G_next_cube_d64_lh_ema_nll_s{seed}_s{seed}.pt" --init-backbone-only
           --tag "_d64_cg2w_ema_nll_s${seed}") ;;
  9) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --d 128 --depth 6 --select survnll --epochs 15
           --init-from "$CK/sts_cache_G_next_cube_d128_lh_ema_nll_s{seed}_s{seed}.pt" --init-backbone-only
           --tag "_d128_cg2w_ema_nll_s${seed}") ;;
  10) args=(--head lh --d 64 --depth 4 --select survnll --xcal 10 --tag "_d64_lh_ema_nll_xcal10_s${seed}") ;;
  11) args=(--head lh --d 64 --depth 4 --select survnll --xcal 100 --tag "_d64_lh_ema_nll_xcal100_s${seed}") ;;
  12) args=(--head lh --d 64 --depth 4 --select survnll --attn joint --tag "_d64_lh_ema_nll_joint_s${seed}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt --ema 0.999 \
    "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
