#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_grid_check
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Prueft Gitter, dtype und Bandzahl ueber ALLE 338 Jahresarchive.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s2d_grid_consistency --config "$CONFIG"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
