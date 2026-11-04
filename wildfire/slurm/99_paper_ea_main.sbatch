#!/bin/bash
#SBATCH --job-name=wf_ea_main
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Hauptprotokoll (d 64, EMA, Auswahl NLL) auf flaechentreuen Ausschnitten (80), sonst
# exakt wie auf den Originalpixeln. Index i: Variante i / 3, Seed i % 3.
#   v0  LH wie 68 v5, 10 Epochen               -> sts_cache_G_ea_next_cube_d64_lh_ema_nll_s<seed>_s<seed>.pt
#   v1  Contagion v2 von Grund auf wie 83 v1    -> sts_cache_G_ea_next_cube_d64_cg2_ema_nll_s<seed>_s<seed>.pt
# Auswertung: 99_paper_ea_main_eval.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
base=(--cache cache_G_ea --train-source cube --cube-dir paper_pixel_ea --surv-dir surv_eval_test_ea
      --target next_patch --steps-per-epoch 3000 --ema 0.999 --d 64 --depth 4 --select survnll)
case $v in
  0) args=("${base[@]}" --head lh --epochs 10 --tag "_d64_lh_ema_nll_s${seed}") ;;
  1) args=("${base[@]}" --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --epochs 15
           --tag "_d64_cg2_ema_nll_s${seed}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --seeds $seed --save-ckpt "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
