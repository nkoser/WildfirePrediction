#!/bin/bash
#SBATCH --job-name=wf_paper_st
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll, Schritt 7: unsere ST-DCTM-Architektur auf den Paper-Labels
# (Tages-CNN -> Gedaechtnis -> Cross-Attention -> Bernstein-Kopf ueber den
# Vorlauf), trainiert mit den vier Paper-dt und mit zufaelligem dt in 1..30.
# Braucht 40_paper_prepare (_L61).
#   0 -> cache_A   1 -> cache_G   2 -> cache_H   3 -> cache_Y
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CACHES=(cache_A cache_G cache_H cache_Y)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.stdctm --config "$CONFIG" \
    --cache "${CACHES[${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
