#!/bin/bash
#SBATCH --job-name=wf_audit_chansem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_chansem
# Stufe 2b: prueft die README-Bandbedeutungen gegen Sommer-/Winterkontrast
# und gegen die deklarierte Kadenz (statisch vs. tatsaechlich veraenderlich).
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s2b_channel_semantics --config "$CONFIG" --year "${REF_YEAR:-2015}"
echo "[$(date -u +%FT%TZ)] exit $?"
