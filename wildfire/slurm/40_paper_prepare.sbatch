#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_paper_prep
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll, Schritt 1: je Array-Task eine Cache-Variante
#   0 -> cache_A   1 -> cache_G   2 -> cache_H   3 -> cache_Y
# liest /windows (14,8 GB) von /firedata und schreibt die letzten LOOKBACK
# Tage nach $EXP/derived_data/paper/<cache>/ (rund 7 GB bei T = 30).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CACHES=(cache_A cache_G cache_H cache_Y)
CACHE=${CACHES[${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}]}
"$PY" -m wildfire.paper.prepare --config "$CONFIG" --cache "$CACHE" --lookback "${LOOKBACK:-30}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
