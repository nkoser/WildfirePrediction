#!/bin/bash
#SBATCH --job-name=wf_contagion_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Zerlegung des Contagion-STS in Zuendung und Ausbreitung auf den zufaelligen
# Test-Stichtagen (surv_eval), gegen den empirischen Persistenz-Anteil je tau.
# Braucht 63_paper_contagion. Anderes Modell: PATTERN="<glob der Checkpoints>" EXTRA="--tag _x"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_eval --config "$CONFIG" ${PATTERN:+--pattern "$PATTERN"} ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
