#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_audit_bt_leak
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 2c: trennt Detektions- von Vorhersagesignal in den
# Helligkeitstemperatur-Kanaelen (ch50-53) ueber den zeitlichen Verlauf
# der Anomalie vor dem Detektionstag.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s2c_bt_leakage --config "$CONFIG" --n-days "${N_DAYS:-30}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
