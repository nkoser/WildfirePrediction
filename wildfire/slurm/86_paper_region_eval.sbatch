#!/bin/bash
#SBATCH --job-name=wf_region_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Regionen-Auswertung, Schritt 2: Survival-Kurven fuer Regionen (erstes Feuer irgendwo in R)
# auf einer zufaelligen Teilmenge der Test-Stichtage. Contagion-STS (Monte Carlo, Mean-Field,
# unabhaengiges Produkt) gegen STS-LH (unabhaengiges Produkt, Ausschnittskopf).
# Checkpoints des Ablationsprotokolls, nur gelesen. Braucht 85_paper_region_events.
# Ausgabe NEU: metrics/paper_region/region_eval<tag>.json (bricht ab, wenn sie existiert).
# Probelauf:  EXTRA="--n 2000 --M 16 --tag _smoke" sbatch 86_paper_region_eval.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.region_eval --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
