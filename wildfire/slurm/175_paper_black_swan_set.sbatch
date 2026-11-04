#!/bin/bash
#SBATCH --job-name=wf_bs_set
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Black Swans: dichte Stichtagsmenge je Ereignis (wildfire/paper/black_swan_dense.py --part set), nur CPU.
# Ausgabe NEU: derived_data/black_swan${TAG}/set.npz, Manifest p40_black_swan_set${TAG}. Danach 176, 177.
#   Rauchtest: TAG=_smoke EXTRA="--event-ids 16 --limit-days 3" sbatch 175_paper_black_swan_set.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.black_swan_dense --config "$CONFIG" --part set --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
