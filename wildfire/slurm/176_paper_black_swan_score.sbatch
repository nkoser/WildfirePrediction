#!/bin/bash
#SBATCH --job-name=wf_bs_score
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Black Swans, GPU (wildfire/paper/black_swan_score.py): Scores je Modell (3 Seeds) auf der dichten Menge
# aus 175 und dem Paper-Test (cache_G_L61). Je Array-Index ein Modell: 0 lh_ref, 1 lh_mono, 2 cg_ref, 3 cg_mono.
# Ausgabe NEU: derived_data/black_swan${TAG}/score_<modell>.npz, paper_<modell>.npz, Manifest p40_black_swan_score_*.
#   Rauchtest: TAG=_smoke EXTRA="--limit-paper 512" sbatch --array=0 176_paper_black_swan_score.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
models=(lh_ref lh_mono cg_ref cg_mono)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.black_swan_score --config "$CONFIG" --model "${models[$i]}" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
