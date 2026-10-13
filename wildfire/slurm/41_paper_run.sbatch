#!/bin/bash
#SBATCH --job-name=wf_paper_run
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll, Schritt 2: Modelle je Cache-Variante trainieren und mit den
# Paper-Metriken auswerten. Braucht 40_paper_prepare.
#   0 -> cache_A   1 -> cache_G   2 -> cache_H   3 -> cache_Y
#   SEEDS="0 1 2"  SETS="paper no_fire no_fire_bt"   uebersteuerbar
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CACHES=(cache_A cache_G cache_H cache_Y)
CACHE=${CACHES[${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}]}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.run --config "$CONFIG" --cache "$CACHE" \
    --seeds ${SEEDS:-0 1 2} --sets ${SETS:-paper no_fire no_fire_bt}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
