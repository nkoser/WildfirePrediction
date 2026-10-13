#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_sanity
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BEWUSST OHNE --partition: reine CPU-Jobs nehmen die Default-Partition.
# Sanity Checks aus Phase 4.2 - Labelkette gegen Rohdaten, Zeitausrichtung,
# Kanalsperre, Splitverteilungen. Exit 4 bei Abweichung.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.sanity --config "$CONFIG" --split "${SPLIT:-temporal}"
echo "[$(date -u +%FT%TZ)] exit $?"
