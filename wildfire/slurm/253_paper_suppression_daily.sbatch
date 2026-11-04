#!/bin/bash
#SBATCH --job-name=wf_suppression_daily
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Taeglicher Bekaempfungsstatus (CWFIS Reported Fires, 252) gegen die Kalibrierung aller Modelle aller Splits
# (wildfire/paper/suppression_daily.py; Scores aus metrics/suppression_splits, Job 5697).
#   sbatch --dependency=afterok:<252-job> 253_paper_suppression_daily.sbatch
# Ausgaben NEU: metrics/suppression_daily${TAG}/{summary.md,groups.csv,coverage.csv,links.npz}
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.suppression_daily --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
