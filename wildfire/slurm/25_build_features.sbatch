#!/bin/bash
#SBATCH --job-name=wf_features
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_features
# BEWUSST OHNE --partition: reiner CPU-Job.
# Merkmale fuer B1 - nur das Zentrumspixel, damit der spaetere Vergleich
# mit B2/B3 den Beitrag des raeumlichen Kontexts isoliert.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.features --config "$CONFIG" --split "${SPLIT:-temporal}" \
    --n-train "${N_TRAIN:-400000}" --n-eval "${N_EVAL:-300000}"
echo "[$(date -u +%FT%TZ)] exit $?"
