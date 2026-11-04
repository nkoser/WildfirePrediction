#!/bin/bash
#SBATCH --job-name=wf_ms_train
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Multi-State-Contagion-STS (paper/multistate.py) von Grund auf, Schleife und Einstellungen exakt wie
# 91_paper_contagion_cause_scratch (d 64, Kern 7, 30 Beobachtungstage, Brenndauer-Kurve, B 32, 10 x 3.000 Schritte,
# EMA 0,999, letzte Gewichte). Kontrollen ohne Rueckkehr existieren: _cause_lam0_scratch / _cause_lam1_scratch.
# Index i: Variante i / 3, Seed i % 3.
#   v0  lambda 0  (nur Likelihood)              -> checkpoints/paper_cause/sts_cache_G_next_cube_cause_ms_lam0_scratch_s<seed>.pt
#   v1  lambda 1  (NFDB-Zuendung + Ausbreitung) -> ..._cause_ms_lam1_scratch_s<seed>.pt
#   sbatch --array=0 170_...sbatch smoke   Rauchtest: 2.000 Stichtage, 20 Schritte, Laden ueber den Wrapper in
#                                          multistate_eval (500 Stichtage) und paper_metrics_ckpt (Tag _ms_smoke)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
W=(-m wildfire.paper.multistate)
CKC=/data/Wildfire/checkpoints/paper_cause
if [ "${1:-}" = smoke ]; then
  "$PY" "${W[@]}" train --config "$CONFIG" --seed 0 --init-from= --epochs 1 --steps-per-epoch 20 --limit 2000 \
      --lam 1 --tag _ms_smoke || exit 1
  "$PY" -m wildfire.paper.multistate_eval --config "$CONFIG" --n 500 --tag _smoke \
      --models "$CKC/sts_cache_G_next_cube_cause_ms_smoke_s0.pt" \
               "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_scratch_s0_s0.pt" || exit 1
  "$PY" "${W[@]}" wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
      --models "sts_ms_smoke:$CKC/sts_cache_G_next_cube_cause_ms_smoke_s0.pt" --tag _ms_smoke
  rc=$?
  echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
fi
case $v in
  0) args=(--lam 0 --tag _ms_lam0_scratch) ;;
  1) args=(--lam 1 --tag _ms_lam1_scratch) ;;
esac
"$PY" "${W[@]}" train --config "$CONFIG" --seed $seed --init-from= --epochs 10 "${args[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
