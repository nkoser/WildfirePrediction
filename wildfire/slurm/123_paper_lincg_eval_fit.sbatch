#!/bin/bash
#SBATCH --job-name=wf_lincg_eval
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Lineare Contagion (121, _abl_lincg Seeds 0-2): Auswertung wie die Ablation in 84 Gruppe 3 und Nachkalibrierung wie
# 109. Alle Aufrufe ueber den Wrapper paper/contagion_linear.py (erkennt den linearen Checkpoint beim Laden).
#   0    surv_eval (100.000 zufaellige Test-Stichtage, --calibrate --subsets) und paper_metrics_ckpt (Paper-Protokoll)
#        -> Manifeste mit Tag _lincg
#   1-3  joint_calib --part fit, Seed 0-2 -> metrics/paper_joint/calib/<ckpt>.json (neue Dateien)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
W=(-m wildfire.paper.contagion_linear)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$i" = 0 ]; then
  m=("sts_abl_lincg:$CK/${G}_abl_lincg_s?_s?.pt")
  "$PY" "${W[@]}" wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --calibrate --subsets \
      --models "${m[@]}" --tag _lincg || exit 1
  "$PY" "${W[@]}" wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
      --models "${m[@]}" --tag _lincg
else
  s=$((i - 1))
  "$PY" "${W[@]}" wildfire.paper.joint_calib --config "$CONFIG" --part fit --calib-dir calib \
      --models "lincg=cg:$CK/${G}_abl_lincg_s${s}_s${s}.pt"
fi
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
