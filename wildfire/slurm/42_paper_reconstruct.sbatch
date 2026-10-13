#!/bin/bash
#SBATCH --job-name=wf_paper_rec
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll, Schritt 3: 3D-U-Net ueber Kanaele x Fensterlaenge x
# Fensterende, Ziel PR-AUC ~0,68 aus dem Paper. Braucht 40_paper_prepare mit
# LOOKBACK=61 TAG=_L61 fuer die jeweilige Variante.
#   0 -> cache_G, Normierung des Caches
#   1 -> cache_A, Normierung des Caches
#   2 -> cache_G, eigene Normierung (Kontrolle)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CACHES=(cache_G cache_A cache_G); NORMS=(cache cache ours)
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.reconstruct --config "$CONFIG" --cache "${CACHES[$i]}" \
    --norm "${NORMS[$i]}" --seeds ${SEEDS:-0} ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
