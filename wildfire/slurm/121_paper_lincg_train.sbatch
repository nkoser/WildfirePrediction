#!/bin/bash
#SBATCH --job-name=wf_lincg
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Lineare Contagion (paper/contagion_linear.py): Kopf wie v26 cg_scratch der Ablation (74: K 7, 30 Beobachtungstage,
# Brennkurve, ohne Warmstart, 10 x 3000 Schritte, EMA 0.999, letzte Gewichte), Encoder linear statt STS-Rumpf
# (d 128 >= 113 Merkmale, kein Rang-Engpass). Seed = Array-Index.
#   sbatch 121_...sbatch          voller Lauf  -> checkpoints/paper/sts_cache_G_next_cube_abl_lincg_s<seed>_s<seed>.pt
#   sbatch --array=0 121_...sbatch smoke        Rauchtest (50 Schritte, Tag _lincg_smoke) und Laden in joint_calib
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
seed=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
if [ "${1:-}" = smoke ]; then
  run=(--epochs 1 --steps-per-epoch 50 --limit-eval 2000 --tag _lincg_smoke)
else
  run=(--epochs 10 --steps-per-epoch 3000 --tag "_abl_lincg_s${seed}")
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_linear train --config "$CONFIG" --cache cache_G --train-source cube \
    --cube-dir paper_pixel --surv-dir surv_eval_test --target next_patch --d 128 --depth 4 --bs 32 --lr 3e-4 \
    --warmup 500 --ema 0.999 --select last --seeds $seed --save-ckpt "${cg[@]}" "${run[@]}"
rc=$?
if [ $rc -eq 0 ] && [ "${1:-}" = smoke ]; then          # Laden ueber den Wrapper: Kalibrierung auf wenigen Val-Stichtagen
  "$PY" -m wildfire.paper.contagion_linear wildfire.paper.joint_calib --config "$CONFIG" --part fit \
      --calib-dir calib_smoke_lincg --n-val 128 --J 4 --M 2 --rows 4096 --cg-line -1 0 --cg-step 0.5 \
      --cg-min-step 0.5 --models "lincg=cg:$CK/sts_cache_G_next_cube_lincg_smoke_s${seed}.pt"
  rc=$?
fi
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
