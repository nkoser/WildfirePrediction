#!/bin/bash
#SBATCH --job-name=wf_nofire
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll (Ziel D+1, cache G) ohne Feuermaske als Eingabe. Das MPCFire-
# Manuskript nennt MOD/MYD14A1 "not included among the model input channels";
# die Caches tragen die Maske als Kanal 0, unser Nachbau nutzt sie (paper55).
#   0    3D-U-Net wie _next_paperhp_w0, Kanaele no_fire  -> rec_cache_G_cache_next_paperhp_w0_nofire
#   1    3D-U-Net wie _next_paperhp_w0, Kanaele no_modis -> rec_cache_G_cache_next_paperhp_w0_nomodis
#   2-4  LH d 64, EMA, Auswahl NLL (wie 68 v5), no_fire, Seed 0-2  -> p15_..._no_fire_cube_d64_lh_ema_nll_s<seed>
#   5-7  dasselbe mit no_modis                                    -> p15_..._no_modis_cube_d64_lh_ema_nll_s<seed>
# Auswertung der U-Nets: 86_paper_nofire_eval. Die LH-Laeufe werten sich selbst im
# Paper-Protokoll aus (mit maskiertem Kanal); Bezug: 68 v5 (paper55).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$i" -lt 2 ]; then
  cs=(no_fire no_modis); rn=(_nofire _nomodis)
  "$PY" -m wildfire.paper.reconstruct --config "$CONFIG" --cache cache_G --norm cache --seeds 0 1 2 \
      --channels "${cs[$i]}" --target next_patch --T 30 --end 1 7 15 30 --bs 512 --lr 1e-4 --epochs 150 \
      --patience 15 --run-name "_next_paperhp_w0${rn[$i]}" ${EXTRA:-}
else
  j=$((i - 2)); seed=$((j % 3)); if [ $j -lt 3 ]; then cs=no_fire; else cs=no_modis; fi
  "$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache cache_G --train-source cube \
      --target next_patch --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt --ema 0.999 \
      --head lh --d 64 --depth 4 --select survnll --channels $cs --tag "_d64_lh_ema_nll_s${seed}" ${EXTRA:-}
fi
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
