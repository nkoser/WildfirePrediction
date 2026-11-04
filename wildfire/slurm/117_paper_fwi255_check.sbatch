#!/bin/bash
#SBATCH --job-name=wf_fwi255
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Bedeutung des Werts 255 in FWI / ISI / DC / BUI (wildfire/paper/fwi255_check.py): Kappung oder NoData?
# Liest den Wuerfel, Ausgabe NEU: Manifest p31_fwi255_check.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.fwi255_check --config "$CONFIG" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
