#!/bin/bash
#SBATCH --job-name=wf_b4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_b4
# BEWUSST OHNE --partition und OHNE GPU: ein MLP auf 218 Merkmalen laeuft
# auf CPU in Minuten. GPU braucht erst B2 mit seinen Faltungen.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.models.mlp_survival --config "$CONFIG" --split "${SPLIT:-temporal}"
echo "[$(date -u +%FT%TZ)] exit $?"
