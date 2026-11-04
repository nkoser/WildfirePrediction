#!/bin/bash
#SBATCH --job-name=wf_paper_imp
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Permutation Importance je Kanal, Kanalgruppe und (STS) Kalendertag auf dem
# Test-Split des Papers, ohne neues Training. Braucht 40_paper_prepare (_L61),
# 54 fuer D+1 und die Checkpoints aus 56 bzw. 50.
#   0-2 -> STS-DCTM Seeds 0-2      3-5 -> ST-DCTM pro Pixel Seeds 0-2
# Zusammenfassen (CSV + Abbildung), danach und ohne Array/GPU:
#   sbatch --array= --gres= --dependency=afterok:<job> --export=ALL,AGGREGATE=1 59_paper_importance.sbatch
#   EXTRA="--repeats 3"   EXTRA="--dts 1 15"   EXTRA="--target cache"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
if [ -n "${AGGREGATE:-}" ]; then
    "$PY" -m wildfire.paper.importance --config "$CONFIG" --cache "${CACHE:-cache_G}" --aggregate ${EXTRA:-}
else
    i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
    KINDS=(sts sts sts pxst pxst pxst)
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m wildfire.paper.importance --config "$CONFIG" --cache "${CACHE:-cache_G}" \
        --kind "${KINDS[$i]}" --index $((i % 3)) ${EXTRA:-}
fi
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
