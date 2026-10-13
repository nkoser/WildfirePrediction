#!/bin/bash
#SBATCH --job-name=wf_audit_geography
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 3b: geographische Plausibilitaet, persistente Hotspots, Saisonalitaet.
# Setzt die Karte burn_count.npy aus Stufe 3 voraus.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s3b_geography --config "$CONFIG" --scratch "${SCRATCH_DIR:-/data/Wildfire/derived_data/audit_scratch}"
echo "[$(date -u +%FT%TZ)] exit $?"
