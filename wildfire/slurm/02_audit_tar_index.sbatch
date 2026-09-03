#!/bin/bash
#SBATCH --job-name=wf_audit_tar_index
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_tar_index
# Stufe 1: liest NUR die tar-Header aller 338 Archive (kein Entpacken),
# leitet daraus Zeitachse, Kadenz, Luecken und Duplikate je Modalitaet ab.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s1_tar_index --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
