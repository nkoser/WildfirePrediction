#!/bin/bash
#SBATCH --job-name=wf_audit_label_def
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_label_def
# Stufe 6: bestimmt die Labelregel der Caches empirisch und quantifiziert,
# wie stark der Feuerkanal im Eingabefenster das Label bereits verraet.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s6_label_def --config "$CONFIG" --cache "${CACHE:-cache_G}" --n-samples "${N_SAMPLES:-200}"
echo "[$(date -u +%FT%TZ)] exit $?"
