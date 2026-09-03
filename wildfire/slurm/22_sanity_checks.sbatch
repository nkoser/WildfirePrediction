#!/bin/bash
#SBATCH --job-name=wf_sanity
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_sanity
# BEWUSST OHNE --partition: reine CPU-Jobs nehmen die Default-Partition.
# Sanity Checks aus Phase 4.2 - Labelkette gegen Rohdaten, Zeitausrichtung,
# Kanalsperre, Splitverteilungen. Exit 4 bei Abweichung.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.sanity --config "$CONFIG" --split "${SPLIT:-temporal}"
echo "[$(date -u +%FT%TZ)] exit $?"
