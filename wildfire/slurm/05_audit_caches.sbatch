#!/bin/bash
#SBATCH --job-name=wf_audit_caches
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_caches
# Stufe 4: HDF5-Fenster-Caches und Sample-Indizes der vier Varianten.
# Struktur, Kanalzahl, Fensterlaenge, Splits, Klassenverteilung,
# NaN-Anteile und Kanal-Statistiken auf einer Zufallsstichprobe.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.audit.s4_caches --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
