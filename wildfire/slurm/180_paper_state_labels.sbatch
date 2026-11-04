#!/bin/bash
#SBATCH --job-name=wf_state_lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Zustaende (U/O/B) und Typen des ersten Feuers fuer die Test-Stichtage (wildfire/paper/state_labels.py). Nur CPU.
#   sbatch 180_...sbatch smoke     3.000 Stichtage, Schluessel aus fire_index -> state_labels/eval_index_smoke.*
#   sbatch 180_...sbatch           alle 100.000, Schluessel aus fire_index und aus den Roh-MCD14A1 (Vergleich)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
rc=0
if [ "${1:-}" = smoke ]; then
  "$PY" -m wildfire.paper.state_labels --config "$CONFIG" --part eval --keys index --max-n 3000 --tag _smoke || rc=$?
else
  for k in index raw; do
    "$PY" -m wildfire.paper.state_labels --config "$CONFIG" --part eval --keys $k || rc=$?
  done
fi
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
