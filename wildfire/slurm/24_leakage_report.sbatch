#!/bin/bash
#SBATCH --job-name=wf_leakage_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_leakage_report
# BEWUSST OHNE --partition: reiner CPU-Job.
# Rendert leakage_report.md deterministisch aus den Stufen-Manifesten.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.leakage_report --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
