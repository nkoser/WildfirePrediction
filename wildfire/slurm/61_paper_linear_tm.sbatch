#!/bin/bash
#SBATCH --job-name=wf_linear_tm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Lineares Transformationsmodell mit cloglog-Link (= Cox PH, glatte Bernstein-
# Basis): Hazard Ratios je +1 SD, multivariabel und univariabel, mit Wald-
# Intervallen; Auswertung auf den Test-Stichtagen von 57_paper_surv_eval.
#   EXTRA="--link logistic"    Odds Ratios (Proportional Odds) statt Hazard Ratios
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.linear_tm --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
