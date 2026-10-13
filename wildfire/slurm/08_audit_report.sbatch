#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_audit_report
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 7: erzeugt data_audit.md und dataset_manifest.json deterministisch
# aus den Stufen-Manifesten. Keine Zahl wird von Hand eingetragen.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s7_report --config "$CONFIG"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
