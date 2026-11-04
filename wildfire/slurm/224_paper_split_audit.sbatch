#!/bin/bash
#SBATCH --job-name=wf_split_audit
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Split-Audit (wildfire/paper/split_audit.py): Raum-Zeit-Leakage und Black-Swan-Verteilung von cache_G,
# cache_G_stratified, cache_G_extreme_fire; baut die Kandidaten year_strat und block_strat.
# Ausgabe NEU: metrics/split_audit${TAG}/, derived_data/splits_v2${TAG}/, Manifest p41_split_audit${TAG}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.split_audit --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
