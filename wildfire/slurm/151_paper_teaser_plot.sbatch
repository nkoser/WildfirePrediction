#!/bin/bash
#SBATCH --job-name=wf_teaser_plot
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Ausbreitungs-Teaser aus den npz von 150 (wildfire/paper/teaser_spread.py), nur CPU.
#   sbatch 151_...sbatch scan <tag> [daten...]               Kandidaten-Ausschnitte und Kennzahlen je Schwelle
#   sbatch 151_...sbatch plot <tag> <weitere plot-Argumente>  Abbildung nach figures/teaser/
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
cmd=${1:?scan|plot}; tag=${2:?tag}; shift 2
D=${DATES:-"2023-05-19 2023-05-27 2023-06-02 2023-08-06 2024-08-05 2025-05-26 2025-07-17"}
if [ "$cmd" = scan ]; then
  [ $# -gt 0 ] && D="$*"
  "$PY" -m wildfire.paper.teaser_spread --config "$CONFIG" --tag "$tag" --dates "$D" scan
else
  "$PY" -m wildfire.paper.teaser_spread --config "$CONFIG" --tag "$tag" --dates "$D" plot "$@"
fi
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
