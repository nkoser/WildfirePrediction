#!/bin/bash
#SBATCH --job-name=wf_audit_report
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_report
# Stufe 7: erzeugt data_audit.md und dataset_manifest.json deterministisch
# aus den Stufen-Manifesten. Keine Zahl wird von Hand eingetragen.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s7_report --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
