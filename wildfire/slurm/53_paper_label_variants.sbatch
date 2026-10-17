#!/bin/bash
#SBATCH --job-name=wf_paper_lbl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
# ---------------------------------------------------------------------------
# Gespeicherte Vorhersagen des Paper-Modells (42_paper_reconstruct) gegen
# andere Test-Labels: Ausschnitt/Mittelzelle an D, D+1, D+2, pixelweise.
#   EXTRA="--runs _paperhp _paperhp_last50"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.label_variants --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
