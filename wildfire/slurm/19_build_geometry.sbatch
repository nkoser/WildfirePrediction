#!/bin/bash
#SBATCH --job-name=wf_geometry
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Zellflaechen, Flaechengewichte und Breitenkanaele. Das Gitter ist in Grad
# gleichmaessig, nicht in Metern - ohne Gewichte sind Pixelmittel nach
# Sueden verzerrt.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.geometry --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
