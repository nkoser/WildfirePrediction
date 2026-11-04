#!/bin/bash
#SBATCH --job-name=wf_extreme_tab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Tabellen aus p31_extreme_eval${TAG} (wildfire/paper/extreme_table.py): Markdown ins Log,
# CSV NEU unter metrics/extreme_eval/. Braucht 115.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.extreme_table --config "$CONFIG" --tag "${TAG:-}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
