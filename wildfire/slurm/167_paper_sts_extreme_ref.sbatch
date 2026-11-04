#!/bin/bash
#SBATCH --job-name=wf_sts_extreme_ref
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Vorhandene Ablations-Checkpoints (74, 3 Seeds) mit derselben Auswertung wie sts_extreme (166):
#   0 abl_lh   1 abl_cg_scratch
# Kopf-Bias aus den Ereignisraten der Wuerfel-Stichtage (wie im Training). Ausgaben NEU:
# derived_data/calib_cache_ext/sts_extreme_ref_<name>*.npz, Manifest p35_sts_extreme_ref_<name>.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
case ${SLURM_ARRAY_TASK_ID:?} in
  0) n=abl_lh ;;
  1) n=abl_cg_scratch ;;
esac
"$PY" -m wildfire.paper.sts_extreme --config "$CONFIG" --eval-ckpt "$CK/sts_cache_G_next_cube_${n}_s?_s?.pt" \
    --tag "_ref${TAGX:-}_${n}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
