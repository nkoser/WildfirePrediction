#!/bin/bash
#SBATCH --job-name=wf_patch_maps
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll als 13 x 13-Karte (wildfire/paper/patch_map_eval.py), Ziel D+1, cache_G -> metrics/patch_maps${TAG}/
#   STAGE=scores  Array: Modell = MODELS[i]
#   STAGE=report  (ohne Array)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
MODELS=(
  "persistence_px"
  "unet3d_px"
  "lh=$CK/sts_cache_G_next_cube_abl_lh_s?_s?.pt"
  "cg_scratch=$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"
  "cg_ell11=$CK/sts_spread_cg_ell11_s?_s?.pt"
  "cg_tfonly=$CK/sts_spread_tfonly_cg_tfonly_s?_s?.pt"
  "cg_sf05=$CK/sts_spread_sf_cg_sf05_s?_s?.pt"
  "cg_sf10=$CK/sts_spread_sf_cg_sf10_s?_s?.pt"
  "cg_sf10big=$CK/sts_spread_sf_cg_sf10big_s?_s?.pt"
  "cg_sfiou05=$CK/sts_spread_sfiou_cg_sfiou05_s?_s?.pt"
  "cg_sfiou20=$CK/sts_spread_sfiou_cg_sfiou20_s?_s?.pt"
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case ${STAGE:?STAGE=scores|report} in
  scores) "$PY" -m wildfire.paper.patch_map_eval scores --config "$CONFIG" --tag "${TAG:-}" --model "${MODELS[${SLURM_ARRAY_TASK_ID:?Array}]}" ${EXTRA:-} ;;
  *)      "$PY" -m wildfire.paper.patch_map_eval report --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-} ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
