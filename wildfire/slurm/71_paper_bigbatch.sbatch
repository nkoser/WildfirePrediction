#!/bin/bash
#SBATCH --job-name=wf_bigbatch
#SBATCH --array=0-26
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Groessere Batches. Alle: d 64, 4 Bloecke, EMA, Auswahl nach Survival-NLL, bis 15 Epochen,
# Cosine-Abklingen. Je Batch-Groesse B gleich viele Stichtage pro Epoche (96.000), Lernrate
# 3e-4 * sqrt(B / 32), Warmup und EMA-Horizont in Stichtagen gleich (16.000 bzw. 32.000):
#   B 32: 3000 Schritte, lr 3e-4,   Warmup 500, EMA 0.999
#   B 128: 750 Schritte, lr 6e-4,   Warmup 125, EMA 0.996
#   B 256: 375 Schritte, lr 8.5e-4, Warmup 63,  EMA 0.992
# Cox und DeepHit vergleichen Stichtage innerhalb des Batches (Risikomenge bzw. Ranking),
# darum je mit B 32 als Referenz in derselben Einstellung.
#   0-2  LH, B 128            -> ..._d64_lh_bs128_s<seed>
#   3-5  LH, B 256            -> ..._d64_lh_bs256_s<seed>
#   6-8  Cox, B 256           -> ..._d64_cox_bs256_s<seed>
#   9-11 DeepHit, B 256       -> ..._d64_deephit_bs256_s<seed>
#  12-14 Cox, B 32            -> ..._d64_cox_bs32_s<seed>
#  15-17 DeepHit, B 32        -> ..._d64_deephit_bs32_s<seed>
#  18-20 LH, B 32, Cosine     -> ..._d64_lh_bs32cos_s<seed>   (Referenz ohne Cosine: _d64_lh_ema_nll)
#  21-23 LH, B 256, X-CAL 1   -> ..._d64_lh_bs256_xcal1_s<seed>
#  24-26 Contagion v2, B 256, Rumpf aus 3-5 -> ..._d64_cg2w_bs256_s<seed>  (braucht 3-5 fertig)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); CK=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube
b32=(--bs 32 --steps-per-epoch 3000 --lr 3e-4 --warmup 500 --ema 0.999)
b128=(--bs 128 --steps-per-epoch 750 --lr 6e-4 --warmup 125 --ema 0.996)
b256=(--bs 256 --steps-per-epoch 375 --lr 8.5e-4 --warmup 63 --ema 0.992)
case $v in
  0) args=(--head lh "${b128[@]}" --tag "_d64_lh_bs128_s${seed}") ;;
  1) args=(--head lh "${b256[@]}" --tag "_d64_lh_bs256_s${seed}") ;;
  2) args=(--head cox "${b256[@]}" --tag "_d64_cox_bs256_s${seed}") ;;
  3) args=(--head deephit "${b256[@]}" --tag "_d64_deephit_bs256_s${seed}") ;;
  4) args=(--head cox "${b32[@]}" --tag "_d64_cox_bs32_s${seed}") ;;
  5) args=(--head deephit "${b32[@]}" --tag "_d64_deephit_bs32_s${seed}") ;;
  6) args=(--head lh "${b32[@]}" --tag "_d64_lh_bs32cos_s${seed}") ;;
  7) args=(--head lh "${b256[@]}" --xcal 1 --tag "_d64_lh_bs256_xcal1_s${seed}") ;;
  8) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve "${b256[@]}"
           --init-from "${CK}_d64_lh_bs256_s{seed}_s{seed}.pt" --init-backbone-only
           --tag "_d64_cg2w_bs256_s${seed}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --d 64 --depth 4 --epochs 15 --select survnll --sched cosine --seeds $seed --save-ckpt \
    "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
