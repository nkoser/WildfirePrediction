#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_audit_alignment
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 5: rekonstruiert Kanalzuordnung, Zeitausrichtung und Labeldefinition
# der Fenster-Caches durch exakten Abgleich mit den Rohrastern.
# Das ist der entscheidende Leakage-Test.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s5_alignment --config "$CONFIG" --cache "${CACHE:-cache_G}" --n-samples "${N_SAMPLES:-4}"
echo "[$(date -u +%FT%TZ)] exit $?"
