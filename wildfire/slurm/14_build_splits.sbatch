#!/bin/bash
#SBATCH --job-name=wf_build_splits
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Definiert und PRUEFT die leakage-sicheren Splits: temporal, raeumlich OOD
# (Ost bzw. British Columbia) und Extremsaison 2023. Schreibt versionierte
# Split-Dateien mit Hash.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.splits --config "$CONFIG" \
    --patch-size "${PATCH:-33}" --horizon-days "${HORIZON:-100}"
echo "[$(date -u +%FT%TZ)] exit $?"
