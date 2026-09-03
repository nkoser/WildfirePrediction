#!/bin/bash
#SBATCH --job-name=wf_landmarks
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_landmarks
# Erzeugt den Landmark-Datensatz (Zeit bis zur naechsten Proxy-Zuendung,
# rechtszensiert bei H). SPLIT waehlt die Splitregel.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.landmarks --config "$CONFIG" --split "${SPLIT:-temporal}" --horizon "${HORIZON:-100}" --label-source "${LABEL_SOURCE:-nfdb}"
echo "[$(date -u +%FT%TZ)] exit $?"
