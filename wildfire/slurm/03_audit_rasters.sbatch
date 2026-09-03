#!/bin/bash
#SBATCH --job-name=wf_audit_rasters
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_rasters
# Stufe 2: extrahiert stichprobenweise einzelne GeoTIFFs in einen SCRATCH
# ausserhalb der Rohdaten und liest CRS, Transform, Aufloesung, dtype,
# NoData, Kompression und Wertebereiche.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s2_rasters --config "$CONFIG" \
    --scratch "${SCRATCH_DIR:-/data/Wildfire/derived_data/audit_scratch}"
echo "[$(date -u +%FT%TZ)] exit $?"
