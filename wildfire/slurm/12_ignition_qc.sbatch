#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_ignition_qc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Kreuzt die abgeleiteten Zuendungen mit Landbedeckung und Lage, um den
# Herbstgipfel zu erklaeren (Verdacht: landwirtschaftliches Abbrennen).
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.ignition_qc --config "$CONFIG"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
