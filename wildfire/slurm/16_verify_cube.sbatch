#!/bin/bash
#SBATCH --job-name=wf_verify_cube
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_verify_cube
# Prueft den Wuerfel byte-genau gegen die Rohdaten und misst die
# Lesegeschwindigkeit fuer Trainings-Patches. Exit 3 bei Abweichung.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.verify_cube --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
