#!/bin/bash
#SBATCH --job-name=wf_build_masks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_build_masks
# Baut Land-, Kanada-, Hotspot- und Brennbarkeitsmaske auf dem MPCFire-Gitter.
# Setzt burn_count.npy aus Audit-Stufe 3 voraus.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.masks --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
