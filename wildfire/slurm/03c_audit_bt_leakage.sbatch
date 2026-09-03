#!/bin/bash
#SBATCH --job-name=wf_audit_bt_leak
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_bt_leak
# Stufe 2c: trennt Detektions- von Vorhersagesignal in den
# Helligkeitstemperatur-Kanaelen (ch50-53) ueber den zeitlichen Verlauf
# der Anomalie vor dem Detektionstag.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s2c_bt_leakage --config "$CONFIG" --n-days "${N_DAYS:-30}"
echo "[$(date -u +%FT%TZ)] exit $?"
