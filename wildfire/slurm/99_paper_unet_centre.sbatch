#!/bin/bash
#SBATCH --job-name=wf_unet_centre
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# 3D-U-Net mit den Paper-Hyperparametern wie _paperhp / _next_paperhp_w0 (B 512, LR 1e-4
# fest, BCE, Auswahl Val-BCE, Patience 15, je Delta t von Grund auf), aber auf dem
# Mittelzellen-Label statt dem Ausschnitt-Label.
#   0  Ziel D,   paper55 (reconstruct_centre)          -> rec_cache_G_cache_centre_paperhp
#   1  Ziel D,   no_fire (reconstruct_centre)          -> rec_cache_G_cache_centre_paperhp_no_fire
#   2  Ziel D+1, paper55 (reconstruct, next_centre)    -> rec_cache_G_cache_nextc_paperhp
#   3  Ziel D+1, no_fire (reconstruct, next_centre)    -> rec_cache_G_cache_nextc_paperhp_nofire
# Tabelle: 99_paper_centre_table.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
cs=(paper55 no_fire paper55 no_fire)
rn=(_centre_paperhp _centre_paperhp_no_fire _nextc_paperhp _nextc_paperhp_nofire)
if [ "$i" -lt 2 ]; then mod=wildfire.paper.reconstruct_centre; tg=cache; ends=(2 8 16 31)
else                     mod=wildfire.paper.reconstruct;        tg=next_centre; ends=(1 7 15 30); fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m $mod --config "$CONFIG" --cache cache_G --norm cache --seeds 0 1 2 \
    --channels "${cs[$i]}" --target $tg --T 30 --end "${ends[@]}" --bs 512 --lr 1e-4 --epochs 150 \
    --patience 15 --run-name "${rn[$i]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
