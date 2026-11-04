#!/bin/bash
#SBATCH --job-name=wf_vision
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Vision-Baselines mit LH-Kopf (paper/vision_backbones.py): U-TAE (Garnot & Chanussot 2021) und ConvLSTM (Shi et al.
# 2015) statt des STS-Rumpfs, sonst exakt das Protokoll der Ablations-Referenz 74 v0 (_abl_lh): Wuerfel-Stichtage,
# Ziel D+1, d 64, Batch 32, lr 3e-4, EMA 0.999, 10 x 3000 Schritte, letzte EMA-Gewichte.
#   Index i: Rumpf = (utae, convlstm)[i / 3], Seed = i % 3 -> sts_cache_G_next_cube_abl_<rumpf>_s<seed>_s<seed>.pt
#   sbatch 130_...sbatch                    voller Lauf
#   sbatch --array=0,3 130_...sbatch smoke  Rauchtest (50 Schritte, Tag _<rumpf>_smoke), danach Laden ueber den
#                                           Wrapper in paper_metrics_ckpt und surv_eval (Manifeste *_<rumpf>_smoke)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
BB=(utae convlstm); bb=${BB[$((i / 3))]}; seed=$((i % 3))
CK=/data/Wildfire/checkpoints/paper
W=(-m wildfire.paper.vision_backbones)
if [ "${1:-}" = smoke ]; then
  run=(--epochs 1 --steps-per-epoch 50 --limit-eval 2000 --tag "_${bb}_smoke")
else
  run=(--epochs 10 --steps-per-epoch 3000 --tag "_abl_${bb}_s${seed}")
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "${W[@]}" train "$bb" --config "$CONFIG" --cache cache_G --train-source cube \
    --cube-dir paper_pixel --surv-dir surv_eval_test --target next_patch --d 64 --depth 4 --bs 32 --lr 3e-4 \
    --warmup 500 --ema 0.999 --select last --seeds $seed --save-ckpt --head lh "${run[@]}"
rc=$?
if [ $rc -eq 0 ] && [ "${1:-}" = smoke ]; then
  m=("sts_${bb}_smoke:$CK/sts_cache_G_next_cube_${bb}_smoke_s${seed}.pt")
  "$PY" "${W[@]}" wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
      --models "${m[@]}" --tag "_${bb}_smoke" && \
  "$PY" "${W[@]}" wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --models "${m[@]}" \
      --tag "_${bb}_smoke"
  rc=$?
fi
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
