#!/bin/bash
#SBATCH --job-name=wf_hyb_fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Hybride Abhaengigkeitsschicht (wildfire/paper/hybrid_dep.py): STS-LH-Raender + Contagion-Kern (von Grund auf),
# beide nach joint_calib (calib/). Fit des Kopplungsniveaus theta (Ringe des Kerns + Brenndauer) auf den Val-Jahren,
# Ziel NLL der bedingten Restzeit. Array-Index = Seed-Paar (lh s / cg_scratch s).
#   sbatch 155_...sbatch          -> metrics/paper_joint/hybrid_calib/hyb_s<seed>.json
#   sbatch --array=0 155_...sbatch smoke
#       Rauchtest aller Teile mit winzigen Groessen -> hybrid_calib_smoke<x>/, Tags _hyb_smoke<x> (SMOKE_TAG=_hyb_smoke<x>)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
s=${SLURM_ARRAY_TASK_ID:-0}
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
pair=(--lh-ckpt "$CK/${G}_abl_lh_s${s}_s${s}.pt" --cg-ckpt "$CK/${G}_abl_cg_scratch_s${s}_s${s}.pt" --seed-name "$s")
run() { "$PY" -m wildfire.paper.hybrid_dep --config "$CONFIG" "${pair[@]}" "$@" || { echo "ABBRUCH bei: $*"; exit 1; }; }
if [ "${1:-}" = smoke ]; then
  st=${SMOKE_TAG:-_hyb_smoke}; hd=(--hyb-dir "hybrid_calib${st#_hyb}")
  run --part fit "${hd[@]}" --fit-landmarks 15 --M 2 --max-evals 6 --line 0 1 2
  run --part cond "${hd[@]}" --n 20 --M 2 --tag "$st"
  run --part shapes "${hd[@]}" --n 64 --M 2 --shapes sq1 sq5 ring9_5 sq13 --tag "$st"
  run --part scenarios "${hd[@]}" --n 64 --M 4 --M-counts 8 --tag "$st"
else
  run --part fit
fi
echo "[$(date -u +%FT%TZ)] exit 0"
