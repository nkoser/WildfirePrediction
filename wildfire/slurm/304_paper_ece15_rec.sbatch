#!/bin/bash
#SBATCH --job-name=wf_ece15_rec
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# ECE/MCE mit 10 vs 15 Bins (Paper Anhang D: M = 15) fuer die gespeicherten rohen Vorhersagen
# des 3D-U-Net-Nachbaus (wildfire/paper/ece15_rec.py) -> metrics/ece15/summary.md.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.ece15_rec --config "$CONFIG" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
