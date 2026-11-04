#!/bin/bash
#SBATCH --job-name=wf_fetch_cwfis_reported
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# CWFIS Reported Fires (taegliche Lageberichte: stage_of_control, response_type) 2010-2026 per WFS
# (wildfire/data/cwfis_reported_fetch.py). Vorhandene Jahresdateien bleiben unangetastet.
#   sbatch 252_fetch_cwfis_reported.sbatch
# Ausgaben NEU: $EXP/artifacts/cwfis_reportedfires/{status_<jahr>.csv.gz, fetch_log.json}
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.data.cwfis_reported_fetch ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
