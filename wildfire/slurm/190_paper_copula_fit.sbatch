#!/bin/bash
#SBATCH --job-name=wf_cop_fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Gemeinsames unbeobachtetes Risiko (wildfire/paper/copula_dep.py): additive Gamma-Frailty auf den kalibrierten
# STS-LH-Pixelkurven (calib/), Raender exakt STS-LH. Fit von copg (Clayton, 1 Parameter) und cop (global + lokal,
# 4 Parameter) auf den Val-Jahren, Ziel NLL der bedingten Restzeit. Array-Index = STS-LH-Seed.
#   sbatch 190_...sbatch          -> metrics/paper_joint/copula_calib/cop_s<seed>.json
#   sbatch --array=0 190_...sbatch smoke
#       Rauchtest aller Teile mit winzigen Groessen -> copula_calib_smoke<x>/, Tags _copula_smoke<x> (SMOKE_TAG=_copula_smoke<x>)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
s=${SLURM_ARRAY_TASK_ID:-0}
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
one=(--lh-ckpt "$CK/${G}_abl_lh_s${s}_s${s}.pt" --seed-name "$s")
run() { "$PY" -m wildfire.paper.copula_dep --config "$CONFIG" "${one[@]}" "$@" || { echo "ABBRUCH bei: $*"; exit 1; }; }
if [ "${1:-}" = smoke ]; then
  st=${SMOKE_TAG:-_copula_smoke}; cd_=(--cop-dir "copula_calib${st#_copula}")
  run --part fit "${cd_[@]}" --fit-landmarks 40 --val-shape-landmarks 64 --M 32 --line 2 32 --golden 1 --max-evals 3
  run --part cond "${cd_[@]}" --n 30 --M 32 --tag "$st"
  run --part shapes "${cd_[@]}" --n 64 --shapes sq1 sq5 ring9_5 sq13 --tag "$st"
  run --part scenarios "${cd_[@]}" --n 64 --M 4 --M-counts 8 --tag "$st"
else
  run --part fit
fi
echo "[$(date -u +%FT%TZ)] exit 0"
