#!/bin/bash
#SBATCH --job-name=wf_fire_base
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Vergleichsbasis nur aus Feuerhistorie + Kalendertag (wildfire/paper/fire_baseline.py), nur CPU.
# Ausgaben NEU: metrics/fire_baseline${TAG}/, Manifest p43_fire_baseline${TAG}.
#   Rauchtest: TAG=_smoke EXTRA="--limit-train 5000" sbatch 158_paper_fire_baseline.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.fire_baseline --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
