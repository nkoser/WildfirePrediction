#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_build_cube
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Baut den dichten Datenwuerfel. IO-gebunden: jedes tar wird genau einmal
# sequentiell gelesen. YEARS setzen, z. B.
#   YEARS="2021 2022 2023" sbatch wildfire/slurm/15_build_cube.sbatch
# BEWUSST OHNE --partition und ohne --gres: reine CPU-Jobs nehmen so die
# Default-Partition und warten nicht in der GPU-Queue. Ein explizites
# --partition=gpu liess Job 5196 ueber zwei Stunden in PENDING haengen,
# obwohl der Knoten 224 CPUs hat. Gleiches Muster wie in AneuG und
# femur-sr-paper.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.cube --config "$CONFIG" \
    --years ${YEARS:-2021 2022 2023} --workers "${WORKERS:-8}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
