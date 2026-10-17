#!/bin/bash
#SBATCH --job-name=wf_paper_seval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BRAUCHT DIE ROHDATEN (MCD14A1, ganzes Jahr) und den Wuerfel.
# ---------------------------------------------------------------------------
# Survival-Auswertung auf zufaelligen Stichtagen der Testjahre 2023-2025:
# Zeit bis zum ersten Feuer in der Mitte / im Ausschnitt, bis 100 Tage;
# KM, KM nach Feuer im Fenster, ST-DCTM pro Pixel, STS-DCTM (Checkpoints).
#   EXTRA="--build"     Stichtage neu ziehen
#   EXTRA="--models sts:/data/Wildfire/checkpoints/paper/sts_..._s*.pt --tag _ft"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
