#!/bin/bash
#SBATCH --job-name=wf_state_ms
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# state_eval (181) zusaetzlich mit der Multi-State-Contagion (170: ms_lam0, ms_lam1) und den Kontrollen ohne
# Rueckkehr (cause_lam0_scratch, cause_lam1_scratch), neben lh, cg, utae, convlstm und den klassischen Basislinien;
# alle Kurvenmodelle kalibriert auf den Val-Stichtagen; je Zustand NLL, C^td, PR/AUC und O/E@7/@30.
#   sbatch 185_...sbatch smoke     3.000 / 2.000 Stichtage, 20 Ziehungen -> state_eval_ms_smoke.json
#   sbatch 185_...sbatch           voll                                 -> state_eval_ms.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CKC=/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause
X=(--models lh cg utae convlstm --extra "ms_lam0=${CKC}_ms_lam0_scratch_s?.pt" "ms_lam1=${CKC}_ms_lam1_scratch_s?.pt"
   "cause_lam0=${CKC}_lam0_scratch_s?.pt" "cause_lam1=${CKC}_lam1_scratch_s?.pt")
if [ "${1:-}" = smoke ]; then
  a=(--max-test 3000 --max-val 2000 --boot 20 --label-tag _smoke --tag _ms_smoke)
else
  a=(--tag _ms)
fi
"$PY" -m wildfire.paper.state_eval --config "$CONFIG" "${X[@]}" "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
