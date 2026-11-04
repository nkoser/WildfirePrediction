#!/bin/bash
#SBATCH --job-name=wf_refc_build
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Eingabefenster an L + k fuer den Neustart in der bedingten Vorhersage (joint_refc --part build), alle
# Test-Stichtage mit erstem Feuer im Ausschnitt an Tag 1..70 (wie 111). Nur CPU und Wuerfel.
# Ausgabe NEU: derived_data/joint_refc/test_{X.npy,meta.npz}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part build --build-name test
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
