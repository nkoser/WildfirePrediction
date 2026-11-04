#!/bin/bash
#SBATCH --job-name=wf_cg_scratch
#SBATCH --array=0-44
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS von Grund auf: zu jedem Lauf, dessen Rumpf aus Logistic Hazard
# kam (--init-from ... --init-backbone-only), das Gegenstueck ohne Warmstart.
# Sonst jeweils exakt die Einstellungen des warmen Laufs; die Ablation nimmt
# wie v26 (cg_scratch) 10 statt 3 Epochen. Index i: Variante i / 3, Seed i % 3.
#   v0  wie 68 v1  (d 128, EMA)                       -> ..._d128_cg2_ema_s<seed>
#   v1  wie 68 v8  (d 64, EMA, Auswahl NLL, 15 Ep.)    -> ..._d64_cg2_ema_nll_s<seed>
#   v2  wie 68 v9  (d 128, EMA, Auswahl NLL, 15 Ep.)   -> ..._d128_cg2_ema_nll_s<seed>
#   v3  wie 71 v8  (d 64, B 256, Cosine)               -> ..._d64_cg2_bs256_s<seed>
#   v4  wie 74 v22 (K 1)                               -> ..._abl_cg_k1_scratch_s<seed>
#   v5  wie 74 v23 (K 3)                               -> ..._abl_cg_k3_scratch_s<seed>
#   v6  wie 74 v24 (7 Beobachtungstage)                -> ..._abl_cg_obs7_scratch_s<seed>
#   v7  wie 74 v25 (geometrische Brenndauer)           -> ..._abl_cg_geo_scratch_s<seed>
#   v8  wie 74 v27 (+ Rate-Kopf, Gewicht 1)            -> ..._abl_cg_rate1_scratch_s<seed>
#   v9  wie 81 v5  (flaechentreu, d 128)               -> sts_cache_G_ea_..._d128_cg2_s<seed>
#   v10-v14 wie v4-v8, flaechentreu (74 mit CACHE=cache_G_ea) -> sts_cache_G_ea_..._abl_cg_*_scratch_s<seed>
# Bereits vorhanden von Grund auf: _d128_cg2 (65), _d64_cg2_ema (68 v4),
# _abl_cg_scratch (74 v26), _d128_cg2long (66), _d128_cg3t/_cg3m (79).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
ema=(--cache cache_G --train-source cube --target next_patch --epochs 10 --steps-per-epoch 3000 --ema 0.999)
big=(--cache cache_G --train-source cube --target next_patch --d 64 --depth 4 --epochs 15 --select survnll
     --sched cosine --bs 256 --steps-per-epoch 375 --lr 8.5e-4 --warmup 63 --ema 0.992)
abl=(--cache cache_G --train-source cube --cube-dir paper_pixel --surv-dir surv_eval_test
     --target next_patch --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999
     --epochs 10 --steps-per-epoch 3000 --select last)
abl_ea=(--cache cache_G_ea --train-source cube --cube-dir paper_pixel_ea --surv-dir surv_eval_test_ea
        --target next_patch --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999
        --epochs 10 --steps-per-epoch 3000 --select last)
ea=(--cache cache_G_ea --train-source cube --cube-dir paper_pixel_ea --surv-dir surv_eval_test_ea
    --target next_patch --d 128 --depth 6 --epochs 10 --steps-per-epoch 3000)
case $v in
  0) args=("${ema[@]}" "${cg[@]}" --d 128 --depth 6 --tag "_d128_cg2_ema_s${seed}") ;;
  1) args=("${ema[@]}" "${cg[@]}" --d 64 --depth 4 --select survnll --epochs 15 --tag "_d64_cg2_ema_nll_s${seed}") ;;
  2) args=("${ema[@]}" "${cg[@]}" --d 128 --depth 6 --select survnll --epochs 15 --tag "_d128_cg2_ema_nll_s${seed}") ;;
  3) args=("${big[@]}" "${cg[@]}" --tag "_d64_cg2_bs256_s${seed}") ;;
  4) args=("${abl[@]}" --head contagion --cg-kernel 1 --cg-obs-days 30 --cg-burn curve --tag "_abl_cg_k1_scratch_s${seed}") ;;
  5) args=("${abl[@]}" --head contagion --cg-kernel 3 --cg-obs-days 30 --cg-burn curve --tag "_abl_cg_k3_scratch_s${seed}") ;;
  6) args=("${abl[@]}" --head contagion --cg-kernel 7 --cg-obs-days 7 --cg-burn curve --tag "_abl_cg_obs7_scratch_s${seed}") ;;
  7) args=("${abl[@]}" --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn geometric --tag "_abl_cg_geo_scratch_s${seed}") ;;
  8) args=("${abl[@]}" "${cg[@]}" --rate-weight 1 --tag "_abl_cg_rate1_scratch_s${seed}") ;;
  9) args=("${ea[@]}" "${cg[@]}" --tag "_d128_cg2_s${seed}") ;;
  10) args=("${abl_ea[@]}" --head contagion --cg-kernel 1 --cg-obs-days 30 --cg-burn curve --tag "_abl_cg_k1_scratch_s${seed}") ;;
  11) args=("${abl_ea[@]}" --head contagion --cg-kernel 3 --cg-obs-days 30 --cg-burn curve --tag "_abl_cg_k3_scratch_s${seed}") ;;
  12) args=("${abl_ea[@]}" --head contagion --cg-kernel 7 --cg-obs-days 7 --cg-burn curve --tag "_abl_cg_obs7_scratch_s${seed}") ;;
  13) args=("${abl_ea[@]}" --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn geometric --tag "_abl_cg_geo_scratch_s${seed}") ;;
  14) args=("${abl_ea[@]}" "${cg[@]}" --rate-weight 1 --tag "_abl_cg_rate1_scratch_s${seed}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --seeds $seed --save-ckpt "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
