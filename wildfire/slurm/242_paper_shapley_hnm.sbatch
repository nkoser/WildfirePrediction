#!/bin/bash
#SBATCH --job-name=wf_shapley_hnm
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Gruppen-Shapley getrennt nach harten / repraesentativen Negativen (FWI-HNM), nur CPU
# (wildfire/paper/shapley_hnm.py). Liest die Scores aus 218 (metrics/shapley_groups${SRC_TAG}/scores_*.npz).
#   sbatch 242_paper_shapley_hnm.sbatch                         Gewichte des Papers -> metrics/shapley_hnm/
#   WEIGHTS=ours TAG=_ours sbatch 242_...                       unsere Rekonstruktion  -> metrics/shapley_hnm_ours/
#   Rauchtest: TAG=_smoke sbatch 242_...
# Ausgaben NEU: metrics/shapley_hnm${TAG}/{summary.md,shapley_hnm.csv}, Manifest p54_shapley_hnm${TAG}
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.shapley_hnm --config "$CONFIG" --src-tag "${SRC_TAG:-}" --weights "${WEIGHTS:-paper}" \
    --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
