#!/bin/bash
#SBATCH --job-name=wf_norm_stats
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_norm_stats
# BEWUSST OHNE --partition: reiner CPU-Job.
# Normalisierung NUR aus Trainingsjahren - eine der Leakage-Pruefungen des Plans.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.norm_stats --config "$CONFIG" --split "${SPLIT:-temporal}"
echo "[$(date -u +%FT%TZ)] exit $?"
