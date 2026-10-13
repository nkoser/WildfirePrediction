#!/bin/bash
#SBATCH --job-name=wf_nfdb_valid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Validiert die abgeleitete Labelkette gegen die kanadische Branddatenbank -
# eine von MODIS unabhaengige Quelle mit Koordinate, Datum, Ursache, Flaeche.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.nfdb_validation --config "$CONFIG" ${SWEEP:+--sweep}
echo "[$(date -u +%FT%TZ)] exit $?"
