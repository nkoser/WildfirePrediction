#!/bin/bash
#SBATCH --job-name=wf_build_events
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_build_events
# Leitet Proxy-Zuendungen (mehrere Definitionen im Vergleich) und
# raumzeitliche Brandereignisse ab. Grundlage fuer Landmark-Datensatz,
# ereignisgruppierte Splits und die Sentinel-Chip-Beschaffung.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.events --config "$CONFIG" --cooldown "${COOLDOWN:-7}" --default-radius "${RADIUS:-1}"
echo "[$(date -u +%FT%TZ)] exit $?"
