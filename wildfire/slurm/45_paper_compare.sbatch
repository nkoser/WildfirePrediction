#!/bin/bash
#SBATCH --job-name=wf_paper_cmp
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll, Schritt 6: fairer Methodenvergleich auf den Paper-Labels
# (Persistenz, 3D-U-Net je dt, HGB je dt, MLP je dt, MLP fuer alle dt,
# 3D-U-Net fuer alle dt). Braucht 40_paper_prepare (_L61) und fuer
# unet3d_dt die Vorhersagen aus 42_paper_reconstruct (_paperdt).
#   0 -> cache_A   1 -> cache_G   2 -> cache_H   3 -> cache_Y
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CACHES=(cache_A cache_G cache_H cache_Y)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.compare --config "$CONFIG" \
    --cache "${CACHES[${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
