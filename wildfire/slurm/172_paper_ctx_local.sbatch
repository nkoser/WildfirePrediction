#!/bin/bash
#SBATCH --job-name=wf_ctx_local
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Regional-Merkmale ohne ytd_ca/anytd_ca (wildfire/paper/ctx_local.py), nur CPU.
# Ausgabe NEU: derived_data/ctx_reg_local/, Manifest p38_ctx_reg_local. Danach 173 (Training).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.ctx_local --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
