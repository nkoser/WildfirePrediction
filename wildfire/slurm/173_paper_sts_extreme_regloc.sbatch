#!/bin/bash
#SBATCH --job-name=wf_sx_regloc
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Wie 166, aber regionale Merkmale ohne die kanadaweiten Spalten (derived_data/ctx_reg_local aus 172):
#   0-2  lh_regloc   LH + --ctx reg (nur lokale Regional-Merkmale)
#   3-5  cg_regloc   Contagion ohne Warmstart + --ctx reg (nur lokale)
# Ausgaben NEU: checkpoints/paper/sts_extreme_<name>_s<seed>_s<seed>.pt, calib_cache_ext/, p35_sts_extreme_<name>_s<seed>.
# Danach 168 mit NAMES="lh_regloc cg_regloc ..." und eigenem TAG.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
case $v in
  0) n=lh_regloc; args=(--head lh --ctx reg) ;;
  1) n=cg_regloc; args=("${cg[@]}" --ctx reg) ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_extreme --config "$CONFIG" --ctx-dir ctx_reg_local \
    --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
