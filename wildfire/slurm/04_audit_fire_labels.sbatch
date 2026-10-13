#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_audit_fire_labels
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 3: VOLLSTAENDIGE Auswertung der Feuer-Labels (MCD14A1, nur 81 MB
# ueber 26 Jahre). Klassenverteilung, Feuer pro Pixel/Jahr, Wiederbrand,
# Ignition vs. Fortsetzung, Ereignis-Zusammenhangskomponenten.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s3_fire_labels --config "$CONFIG" \
    --scratch "${SCRATCH_DIR:-/data/Wildfire/derived_data/audit_scratch}"
echo "[$(date -u +%FT%TZ)] exit $?"
