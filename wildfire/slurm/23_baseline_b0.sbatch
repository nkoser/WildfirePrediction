#!/bin/bash
#SBATCH --job-name=wf_b0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_b0
# BEWUSST OHNE --partition: reiner CPU-Job.
# B0 Klimatologie - braucht weder Wuerfel noch GPU, nur die Zuendungen.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.models.climatology --config "$CONFIG" --split "${SPLIT:-temporal}" --label-source "${LABEL_SOURCE:-nfdb}"
echo "[$(date -u +%FT%TZ)] exit $?"
