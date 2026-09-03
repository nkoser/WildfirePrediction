#!/bin/bash
#SBATCH --job-name=wf_fire_index
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_fire_index
# Duenner Index aller Feuertage und Zuendungen je Pixel (CSR).
# Grundlage der Survival-Labels: "naechste Zuendung nach t" per binaerer
# Suche, und "brennt gerade" fuer den Landmark-Ausschluss.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.fire_index --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
