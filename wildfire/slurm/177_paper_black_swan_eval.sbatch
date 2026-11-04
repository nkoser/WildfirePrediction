#!/bin/bash
#SBATCH --job-name=wf_bs_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Black Swans: Auswertung der dichten Menge und des Paper-Protokolls (black_swan_dense.py --part eval), nur CPU.
# Ausgabe NEU: metrics/black_swan_dense${TAG}/, Manifest p40_black_swan_dense${TAG}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.black_swan_dense --config "$CONFIG" --part eval --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
