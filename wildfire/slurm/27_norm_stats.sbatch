#!/bin/bash
#SBATCH --job-name=wf_norm_stats
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BEWUSST OHNE --partition: reiner CPU-Job.
# Normalisierung NUR aus Trainingsjahren - eine der Leakage-Pruefungen des Plans.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.norm_stats --config "$CONFIG" --split "${SPLIT:-temporal}"
echo "[$(date -u +%FT%TZ)] exit $?"
