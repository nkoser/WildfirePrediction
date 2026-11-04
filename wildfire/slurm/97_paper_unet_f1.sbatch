#!/bin/bash
#SBATCH --job-name=wf_unet_f1
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# 3D-U-Net mit der Abbruchregel des Papers (Early Stopping auf Val-F1, reconstruct_f1),
# sonst wie _paperhp / _next_paperhp_w0 (B 512, LR 1e-4 fest, BCE, je Delta t von Grund auf).
#   0-2  Ziel D   (cache),      paper55 / no_fire / no_modis -> rec_cache_G_cache_paperhp_f1{,_no_fire,_no_modis}
#   3-5  Ziel D+1 (next_patch), paper55 / no_fire / no_modis -> rec_cache_G_cache_next_paperhp_f1{,_nofire,_nomodis}
# Tabelle: 96_paper_dayd_table.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
cs=(paper55 no_fire no_modis paper55 no_fire no_modis)
if [ "$i" -lt 3 ]; then
  tg=cache; ends=(2 8 16 31); rn=(_paperhp_f1 _paperhp_f1_no_fire _paperhp_f1_no_modis)
else
  tg=next_patch; ends=(1 7 15 30); rn=(_next_paperhp_f1 _next_paperhp_f1_nofire _next_paperhp_f1_nomodis)
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.reconstruct_f1 --config "$CONFIG" --cache cache_G --norm cache --seeds 0 1 2 \
    --channels "${cs[$i]}" --target $tg --T 30 --end "${ends[@]}" --bs 512 --lr 1e-4 --epochs 150 \
    --patience 15 --run-name "${rn[$((i % 3))]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
