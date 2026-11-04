#!/bin/bash
#SBATCH --job-name=wf_teaser_rows
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Folienzeilen beobachtet | STS-LH | Contagion-STS mit gemeinsamen Beispielpixeln und Legende mit Wasser
# (teaser_rows.py) aus den Karten von 150. Ausgabe: figures/teaser_rows (bricht ab, wenn vorhanden).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.teaser_rows --config "$CONFIG"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
