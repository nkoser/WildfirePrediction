#!/bin/bash
#SBATCH --job-name=wf_features
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BEWUSST OHNE --partition: reiner CPU-Job.
# Merkmale fuer B1 - nur das Zentrumspixel, damit der spaetere Vergleich
# mit B2/B3 den Beitrag des raeumlichen Kontexts isoliert.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.features --config "$CONFIG" --split "${SPLIT:-temporal}" \
    --n-train "${N_TRAIN:-400000}" --n-eval "${N_EVAL:-300000}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
