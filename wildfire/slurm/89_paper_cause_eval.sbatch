#!/bin/bash
#SBATCH --job-name=wf_cause_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Auswertung der Ursachen-Zerlegung (paper/contagion_cause_eval.py): Zuordnung Zuendung/Ausbreitung auf
# Test-Ereignissen mit Label und Vorhersage der ersten NFDB-Zuendung (Ausschnitt, Mitte), gegen
# _abl_cg und _abl_lh. Braucht 85, 87 (eval) und 88. Alle Checkpoints nur gelesen.
# Ausgabe NEU: metrics/paper_cause/cause_eval<tag>.json (bricht ab, wenn sie existiert).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_cause_eval --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
