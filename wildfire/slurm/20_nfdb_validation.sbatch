#!/bin/bash
#SBATCH --job-name=wf_nfdb_valid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_nfdb_valid
# Validiert die abgeleitete Labelkette gegen die kanadische Branddatenbank -
# eine von MODIS unabhaengige Quelle mit Koordinate, Datum, Ursache, Flaeche.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
"$PY" -m wildfire.data.nfdb_validation --config "$CONFIG" ${SWEEP:+--sweep}
echo "[$(date -u +%FT%TZ)] exit $?"
