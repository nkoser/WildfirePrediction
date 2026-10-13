#!/bin/bash
#SBATCH --job-name=wf_landmarks
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Erzeugt den Landmark-Datensatz (Zeit bis zur naechsten Proxy-Zuendung,
# rechtszensiert bei H). SPLIT waehlt die Splitregel.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.landmarks --config "$CONFIG" --split "${SPLIT:-temporal}" --horizon "${HORIZON:-100}" --label-source "${LABEL_SOURCE:-nfdb}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
