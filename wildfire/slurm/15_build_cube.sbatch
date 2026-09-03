#!/bin/bash
#SBATCH --job-name=wf_build_cube
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_build_cube
# Baut den dichten Datenwuerfel. IO-gebunden: jedes tar wird genau einmal
# sequentiell gelesen. YEARS setzen, z. B.
#   YEARS="2021 2022 2023" sbatch wildfire/slurm/15_build_cube.sbatch
# BEWUSST OHNE --partition und ohne --gres: reine CPU-Jobs nehmen so die
# Default-Partition und warten nicht in der GPU-Queue. Ein explizites
# --partition=gpu liess Job 5196 ueber zwei Stunden in PENDING haengen,
# obwohl der Knoten 224 CPUs hat. Gleiches Muster wie in AneuG und
# femur-sr-paper.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.cube --config "$CONFIG" \
    --years ${YEARS:-2021 2022 2023} --workers "${WORKERS:-8}"
echo "[$(date -u +%FT%TZ)] exit $?"
