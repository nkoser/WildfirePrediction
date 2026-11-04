#!/bin/bash
#SBATCH --job-name=wf_suppression_splits
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Bekaempfung / Bevoelkerung: Kalibrierung auf allen Test-Stichproben aller vier Splits
# (wildfire/paper/suppression_splits.py). Config mit Overlay (norm_stats der Split-Caches).
#   sbatch 248_paper_suppression_splits.sbatch
#   Rauchtest: TAG=_smoke EXTRA="--limit 300" sbatch 248_...
# Ausgaben NEU: metrics/suppression_splits${TAG}/{summary.md,groups.csv,scores.npz}, Manifest p55_suppression_splits${TAG}
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.suppression_splits --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
