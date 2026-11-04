#!/bin/bash
#SBATCH --job-name=wf_label_values
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
# ---------------------------------------------------------------------------
# Rasterwerte von MCD14A1 an den Tagen D+1 der Cache-Stichproben und D+1-Labels unter den
# Regeln > 0 / >= 7 / >= 8 / == 9 (Paper: nur 9 positiv) -> metrics/label_values/summary.md
# (wildfire/paper/label_values.py, Manifest p12_label_values_<cache>).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.label_values --config "$CONFIG" --cache "${CACHE:-cache_G}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
