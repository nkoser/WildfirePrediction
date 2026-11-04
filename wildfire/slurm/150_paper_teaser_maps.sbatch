#!/bin/bash
#SBATCH --job-name=wf_teaser_map
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Dichte Pixel-CIF fuer JEDEN Tag tau = 1..30 (risk_map.py unveraendert, nur --taus) an Stichtagen kurz
# nach dem Ausbruch grosser Feuer 2023-2025, fuer den Ausbreitungs-Teaser (teaser_spread.py, 151).
#   Task 0  Contagion-STS von Grund auf (abl_cg_scratch, 3 Seeds)  -> risk_teaser_cgs_<datum>.npz
#   Task 1  STS-LH (abl_lh, 3 Seeds)                              -> risk_teaser_lh_<datum>.npz
#   DATES="..." uebersteuert die Stichtage; smoke als erstes Argument: ein Datum
# Bricht ab, wenn eine Ausgabe schon existiert (nichts ueberschreiben).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CK=/data/Wildfire/checkpoints/paper
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TAG=teaser_cgs; PAT="$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt" ;;
  1) TAG=teaser_lh;  PAT="$CK/sts_cache_G_next_cube_abl_lh_s?_s?.pt" ;;
esac
DATES=${DATES:-"2023-05-19 2023-05-27 2023-06-02 2023-08-06 2024-08-05 2025-05-26 2025-07-17"}
if [ "${1:-}" = smoke ]; then DATES="2025-05-26"; TAG="${TAG}_smoke"; fi
for d in $DATES; do
  f="$EXP/figures/risk_maps/risk_${TAG}_${d}.npz"
  [ -e "$f" ] && { echo "ABBRUCH: $f existiert"; exit 3; }
done
TAUS=$(seq -s, 1 30)
"$PY" -m wildfire.paper.risk_map --config "$CONFIG" --pattern "$PAT" --dates $DATES --taus "$TAUS" --tag "$TAG"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
