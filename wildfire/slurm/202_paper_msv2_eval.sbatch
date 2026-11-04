#!/bin/bash
#SBATCH --job-name=wf_msv2_ev
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung Multi-State v2 (wildfire/paper/msv2_eval.py): Index 0 zufaellige Test-Stichtage (gesamt, Strata,
# Typ-AUC, Zuordnung; gepaart gegen lhb10 = _abl_lhbce10), Index 1 FireMPC-Protokoll.
#   SMOKE=1 sbatch --array=0-1 202_...sbatch   kleine Teilmengen, Smoke-Checkpoints -> msv2_<teil>_smoke.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
C=/data/Wildfire/checkpoints
V=$C/paper_msv2/sts_cache_G_next_cube_msv2
parts=(surv paper)
if [ -n "${SMOKE:-}" ]; then
  M=("lhb10=$C/paper/sts_cache_G_next_cube_abl_lhbce10_s0_s0.pt" "msv2=${V}_smoke_msv2_s0.pt" "lhf=${V}_smoke_lhf_s0.pt"
     "ms_lam0=$C/paper_cause/sts_cache_G_next_cube_cause_ms_lam0_scratch_s0.pt")
  if [ "$i" -eq 0 ]; then a=(--max-test 2000 --max-val 2000 --boot 20 --n-attr 500 --label-tag _smoke --feat-tag _smoke --tag "${STAG:-_smoke}")
  else a=(--feat-tag _smoke --tag "${STAG:-_smoke}"); fi
else
  M=("lhb10=$C/paper/sts_cache_G_next_cube_abl_lhbce10_s?_s?.pt" "lh=$C/paper/sts_cache_G_next_cube_abl_lh_s?_s?.pt"
     "lhb=${V}_lhb_s?.pt" "lhf=${V}_lhf_s?.pt" "msv2=${V}_msv2_s?.pt" "msv2t=${V}_msv2t_s?.pt" "msf=${V}_msf_s?.pt"
     "msb=${V}_msb_s?.pt" "ms_lam0=$C/paper_cause/sts_cache_G_next_cube_cause_ms_lam0_scratch_s?.pt"
     "cg=$C/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt")
  a=(--tag "${TAG:-}")
fi
"$PY" -m wildfire.paper.msv2_eval "${parts[$i]}" --config "$CONFIG" --models "${M[@]}" --ref lhb10 "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
