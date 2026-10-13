#!/bin/bash
#SBATCH --job-name=wf_b1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BEWUSST OHNE --partition: reiner CPU-Job.
# B1 tabellarisch - logistische Regression und Gradient Boosting je Horizont.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.models.tabular --config "$CONFIG" --split "${SPLIT:-temporal}" --prior-source "${PRIOR:-val}"
echo "[$(date -u +%FT%TZ)] exit $?"
