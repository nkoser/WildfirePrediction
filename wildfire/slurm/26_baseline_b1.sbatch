#!/bin/bash
#SBATCH --job-name=wf_b1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_b1
# BEWUSST OHNE --partition: reiner CPU-Job.
# B1 tabellarisch - logistische Regression und Gradient Boosting je Horizont.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.models.tabular --config "$CONFIG" --split "${SPLIT:-temporal}" --prior-source "${PRIOR:-val}"
echo "[$(date -u +%FT%TZ)] exit $?"
