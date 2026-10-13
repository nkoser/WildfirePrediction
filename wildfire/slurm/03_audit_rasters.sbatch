#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_audit_rasters
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 2: extrahiert stichprobenweise einzelne GeoTIFFs in einen SCRATCH
# ausserhalb der Rohdaten und liest CRS, Transform, Aufloesung, dtype,
# NoData, Kompression und Wertebereiche.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s2_rasters --config "$CONFIG" \
    --scratch "${SCRATCH_DIR:-/data/Wildfire/derived_data/audit_scratch}"
echo "[$(date -u +%FT%TZ)] exit $?"
