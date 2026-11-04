#!/bin/bash
#SBATCH --job-name=wf_reg_shift
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Ursachensuche fuer den Schaden durch --ctx reg (wildfire/paper/reg_shift.py), nur CPU: Regional-Merkmale
# im Training (Wuerfel) gegen Val/Test je Jahr, Verteilung und Rate@30 je Trainings-Dezil.
# Ausgabe NEU: metrics/reg_shift${TAG}/, Manifest p37_reg_shift${TAG}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.reg_shift --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
