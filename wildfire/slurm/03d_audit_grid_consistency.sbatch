#!/bin/bash
#SBATCH --job-name=wf_grid_check
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_grid_check
# Prueft Gitter, dtype und Bandzahl ueber ALLE 338 Jahresarchive.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s2d_grid_consistency --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
