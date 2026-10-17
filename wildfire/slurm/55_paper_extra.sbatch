#!/bin/bash
#SBATCH --job-name=wf_paper_extra
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BRAUCHT DIE ROHDATEN (Feuer-Pixeltage aus MCD14A1) und den Wuerfel.
# ---------------------------------------------------------------------------
# Zusaetzliche Trainingsstichproben im Stil des Papers aus 2000-2019:
# Positive = Feuer in der Mittelzelle an D, Negative mit Puffer, je zur Haelfte
# zufaellig und nach FWI-HNM. Braucht 43_paper_hnm (p3_hnm_score).
#   EXTRA="--n-pos 200 --n-neg 400 --n-cand 20000 --out-name paper_extra_test"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.paper_extra --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
