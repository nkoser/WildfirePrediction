#!/bin/bash
#SBATCH --job-name=wf_black_swan
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Black Swans (wildfire/paper/black_swan.py), nur CPU: extremste Feuerereignisse 2000-2025 aus fire_index
# (Block x Fenster ueber dem lokalen Rekord 2000-2019) und Modellguete der Ensembles aus calib_cache_ext
# auf den surv_eval_test-Stichtagen in den Test-Ereignissen.
# Ausgabe NEU: metrics/black_swan${TAG}/, Manifest p39_black_swan${TAG}.
#   Rauchtest: TAG=_smoke EXTRA="--models lh_ref" sbatch 174_paper_black_swan.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.black_swan --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
