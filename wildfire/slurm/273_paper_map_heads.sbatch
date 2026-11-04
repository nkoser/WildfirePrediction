#!/bin/bash
#SBATCH --job-name=wf_map_heads
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Pixel-CIF tau = 1..30 an den drei Teaser-Stichtagen fuer weitere Koepfe (Ablation, temporaler Split,
# 3 Seeds, gleiches Protokoll wie abl_lh / abl_cg_scratch in 150), fuer map_calib_compare (275).
#   Task 0 DCTM -> risk_teaser_dctm_<datum>.npz, 1 Cox -> teaser_cox, 2 DeepHit -> teaser_deephit
# Bricht ab, wenn eine Ausgabe schon existiert (nichts ueberschreiben).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CK=/data/Wildfire/checkpoints/paper
HEADS=(dctm cox deephit)
H=${HEADS[${SLURM_ARRAY_TASK_ID:-0}]}
TAG=teaser_$H
DATES="2023-05-19 2023-06-02 2025-05-26"
for d in $DATES; do
  f="$EXP/figures/risk_maps/risk_${TAG}_${d}.npz"
  [ -e "$f" ] && { echo "ABBRUCH: $f existiert"; exit 3; }
done
"$PY" -m wildfire.paper.risk_map --config "$CONFIG" --pattern "$CK/sts_cache_G_next_cube_abl_${H}_s?_s?.pt" \
  --dates $DATES --taus "$(seq -s, 1 30)" --tag "$TAG"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
