#!/bin/bash
#SBATCH --job-name=wf_spread_mech
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Zuendmechanismen aus MODIS (wildfire/paper/spread_mechanisms.py): kooperative Zuendung (gamma), Verstaerkung
# ueber das Feueralter, Frontgroesse. Reine Zaehlung (numpy), keine GPU. Ausgabe NEU: metrics/spread_mechanisms_<split>.json/.md
#   SPLIT=train (Standard)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.spread_mechanisms --config "$CONFIG" --split "${SPLIT:-train}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
