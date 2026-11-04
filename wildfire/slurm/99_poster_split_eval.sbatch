#!/bin/bash
#SBATCH --job-name=wf_poster_split_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# U-Net-Vorhersagen cache_G (Label Cache + next_patch) auf Poster-Testjahren 2023-2024 neu auswerten
# (wildfire/paper/poster_split_eval.py). Ausgabe NEU: metrics/poster_split_eval/.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.poster_split_eval --config "$CONFIG"
rc=$?; echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
