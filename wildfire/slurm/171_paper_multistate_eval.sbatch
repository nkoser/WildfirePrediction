#!/bin/bash
#SBATCH --job-name=wf_ms_eval
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung der Multi-State-Contagion (170) gegen die Kontrollen ohne Rueckkehr, alles ueber den Wrapper
# paper/multistate.py (erkennt die Multi-State-Checkpoints beim Laden, alle anderen laden wie bisher).
#   0  surv_eval (100.000 zufaellige Test-Stichtage, --calibrate --subsets) und paper_metrics_ckpt (MPCFire)
#      fuer ms_lam0, ms_lam1, Kontrollen cause_lam0_scratch, cause_lam1_scratch, abl_cg_scratch, abl_lh -> Tag _ms
#   1  multistate_eval: Ursachen-Zuordnung, Ursachen-CIFs, Population nach Zustand -> metrics/paper_multistate/ms_eval.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
CKC=/data/Wildfire/checkpoints/paper_cause
G=sts_cache_G_next_cube
W=(-m wildfire.paper.multistate)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
m=("sts_ms_lam0:$CKC/${G}_cause_ms_lam0_scratch_s?.pt" "sts_ms_lam1:$CKC/${G}_cause_ms_lam1_scratch_s?.pt"
   "sts_cause_lam0_scr:$CKC/${G}_cause_lam0_scratch_s?.pt" "sts_cause_lam1_scr:$CKC/${G}_cause_lam1_scratch_s?.pt"
   "sts_abl_cg_scratch:$CK/${G}_abl_cg_scratch_s?_s?.pt" "sts_abl_lh:$CK/${G}_abl_lh_s?_s?.pt")
if [ "$i" = 0 ]; then
  "$PY" "${W[@]}" wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --calibrate --subsets \
      --models "${m[@]}" --tag _ms || exit 1
  "$PY" "${W[@]}" wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
      --models "${m[@]}" --tag _ms
else
  "$PY" -m wildfire.paper.multistate_eval --config "$CONFIG"
fi
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
