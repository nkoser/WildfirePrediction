#!/bin/bash
#SBATCH --job-name=wf_audit_inventory
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 0: Dateibaum, Groessen, Formate. Reine Standardbibliothek.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s0_inventory --config "$CONFIG"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
