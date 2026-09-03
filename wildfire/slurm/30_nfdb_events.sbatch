#!/bin/bash
#SBATCH --job-name=wf_nfdb_events
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_nfdb_events
# BEWUSST OHNE --partition: reiner CPU-Job.
# Echte Zuendungen aus der kanadischen Branddatenbank als Labelquelle.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.nfdb_events --config "$CONFIG" \
    --min-size-ha "${MIN_SIZE_HA:-1.0}"
echo "[$(date -u +%FT%TZ)] exit $?"
