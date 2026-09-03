#!/bin/bash
#SBATCH --job-name=wf_audit_alignment
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_alignment
# Stufe 5: rekonstruiert Kanalzuordnung, Zeitausrichtung und Labeldefinition
# der Fenster-Caches durch exakten Abgleich mit den Rohrastern.
# Das ist der entscheidende Leakage-Test.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s5_alignment --config "$CONFIG" --cache "${CACHE:-cache_G}" --n-samples "${N_SAMPLES:-4}"
echo "[$(date -u +%FT%TZ)] exit $?"
