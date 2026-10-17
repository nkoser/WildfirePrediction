#!/bin/bash
#SBATCH --job-name=wf_paper_next
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
# ---------------------------------------------------------------------------
# Labels fuer D+1 je Stichprobe (Ausschnitt, Mittelzelle) aus MCD14A1, fuer
# 42_paper_reconstruct --target next_patch. Braucht 40_paper_prepare (_L61).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.next_labels --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
