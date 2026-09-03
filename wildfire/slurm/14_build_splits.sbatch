#!/bin/bash
#SBATCH --job-name=wf_build_splits
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_build_splits
# Definiert und PRUEFT die leakage-sicheren Splits: temporal, raeumlich OOD
# (Ost bzw. British Columbia) und Extremsaison 2023. Schreibt versionierte
# Split-Dateien mit Hash.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.splits --config "$CONFIG" \
    --patch-size "${PATCH:-33}" --horizon-days "${HORIZON:-100}"
echo "[$(date -u +%FT%TZ)] exit $?"
