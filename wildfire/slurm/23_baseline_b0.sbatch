#!/bin/bash
#SBATCH --job-name=wf_b0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BEWUSST OHNE --partition: reiner CPU-Job.
# B0 Klimatologie - braucht weder Wuerfel noch GPU, nur die Zuendungen.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.models.climatology --config "$CONFIG" --split "${SPLIT:-temporal}" --label-source "${LABEL_SOURCE:-nfdb}"
echo "[$(date -u +%FT%TZ)] exit $?"
