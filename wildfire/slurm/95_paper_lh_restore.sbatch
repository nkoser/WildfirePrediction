#!/bin/bash
#SBATCH --job-name=wf_lh_restore
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Wiederherstellung: 85 (Job 4624, Index 2-7) hat die paper55-Checkpoints von 68 v5
# (sts_cache_G_next_cube_d64_lh_ema_nll_s<seed>_s<seed>.pt) ueberschrieben, weil der
# Checkpoint-Name den Kanalsatz nicht trug (seit 2026-10-22 behoben). Dort liegen jetzt
# die no_modis-Modelle (gesichert als sts_cache_G_next_no_modis_cube_..._s<seed>_s<seed>.pt).
# Neu trainiert mit exakt den Einstellungen von 68 v5, unter neuem Tag _rst:
#   0-2  paper55, Seed 0-2  -> sts_cache_G_next_cube_d64_lh_ema_nll_rst_s<seed>_s<seed>.pt
#   3-5  no_fire, Seed 0-2  -> sts_cache_G_next_no_fire_cube_d64_lh_ema_nll_rst_s<seed>_s<seed>.pt
# Erst nach dem Vergleich mit p15_..._d64_lh_ema_nll_s<seed> (2026-10-20) werden die
# paper55-Modelle an den alten Namen kopiert; die Original-Manifeste bleiben unberuehrt.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); if [ "$i" -lt 3 ]; then cs=paper55; else cs=no_fire; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache cache_G --train-source cube \
    --target next_patch --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt --ema 0.999 \
    --head lh --d 64 --depth 4 --select survnll --channels $cs --tag "_d64_lh_ema_nll_rst_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
