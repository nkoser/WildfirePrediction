#!/bin/bash
#SBATCH --job-name=wf_audit_fire_labels
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_fire_labels
# Stufe 3: VOLLSTAENDIGE Auswertung der Feuer-Labels (MCD14A1, nur 81 MB
# ueber 26 Jahre). Klassenverteilung, Feuer pro Pixel/Jahr, Wiederbrand,
# Ignition vs. Fortsetzung, Ereignis-Zusammenhangskomponenten.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s3_fire_labels --config "$CONFIG" \
    --scratch "${SCRATCH_DIR:-/data/Wildfire/derived_data/audit_scratch}"
echo "[$(date -u +%FT%TZ)] exit $?"
