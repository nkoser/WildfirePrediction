#!/bin/bash
#SBATCH --job-name=wf_patch_burn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Karte "brennt an Tag t" aus Contagion-Checkpoints ohne Training (wildfire/paper/patch_map_burn.py)
#   -> metrics/patch_maps${TAG}/scores_<modell>_burn.npz und _burnobs.npz; Bericht danach mit 293 STAGE=report
#   Array: Modell = MODELS[i]
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
MODELS=(
  "cg_scratch=$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"
  "cg_ell11=$CK/sts_spread_cg_ell11_s?_s?.pt"
  "cg_tfonly=$CK/sts_spread_tfonly_cg_tfonly_s?_s?.pt"
  "cg_sf10big=$CK/sts_spread_sf_cg_sf10big_s?_s?.pt"
  "cg_sfiou20=$CK/sts_spread_sfiou_cg_sfiou20_s?_s?.pt"
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.patch_map_burn --config "$CONFIG" --tag "${TAG:-}" --model "${MODELS[${SLURM_ARRAY_TASK_ID:?Array}]}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
