#!/bin/bash
#SBATCH --job-name=wf_sentinel_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_sentinel_probe
# Stufe 8: prueft, ob echte Sentinel-Bilddaten als zusaetzliche Modalitaet
# beschaffbar sind - Verfuegbarkeit, Wolkenlage, effektive Wiederholrate,
# Lesbarkeit per HTTP-Range. Braucht Netzzugang vom Rechenknoten.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s8_sentinel_probe --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
