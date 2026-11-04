#!/bin/bash
#SBATCH --job-name=wf_map_calcache
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Hazards der Teaser-Modelle auf den surv_eval-Stichtagen (Val 2020-2022, Test 2023-2025) und deren
# Kalibrierungsauswertung mit calib_eval.py (unveraendert). Je Modell ein eigener --tag, damit jeder
# Cache eine eigene Datei bekommt: derived_data/calib_cache/sts_teasermap_<m>_sts.npz
# Bricht ab, wenn der Cache schon existiert (nichts ueberschreiben).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CK=/data/Wildfire/checkpoints/paper
M=(lh cg_scratch dctm cox deephit)
m=${M[${SLURM_ARRAY_TASK_ID:-0}]}
TAG="_teasermap_${m}_sts"
[ -e "$EXP/derived_data/calib_cache/sts${TAG}.npz" ] && { echo "ABBRUCH: Cache sts${TAG}.npz existiert"; exit 3; }
"$PY" -m wildfire.paper.calib_eval --config "$CONFIG" --models "sts:$CK/sts_cache_G_next_cube_abl_${m}_s?_s?.pt" --tag "$TAG"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
