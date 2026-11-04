#!/bin/bash
#SBATCH --job-name=wf_unet_weather
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# 3D-U-Net-Nachbau wie 211, Kanalsatz weather (25 Kanaele, in channels.py vorhanden):
#   0  weather  Ziel D    (end 2 8 16 31)   -> p2_reconstruct_cache_G_cache_paperhp_weather
#   1  weather  Ziel D+1  (end 1 7 15 30)   -> p2_reconstruct_cache_G_cache_next_paperhp_w0_weather
# 3 Seeds je dt. Ausgaben NEU.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
if (( i == 0 )); then
  tgt=(--end 2 8 16 31); rn="_paperhp_weather${SFX:-}"
else
  tgt=(--target next_patch --end 1 7 15 30); rn="_next_paperhp_w0_weather${SFX:-}"
fi
if [[ -e "$EXP/data_manifests/p2_reconstruct_cache_G_cache${rn}.json" ]]; then
  echo "ABBRUCH: Manifest p2_reconstruct_cache_G_cache${rn}.json existiert schon"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.reconstruct --config "$CONFIG" --cache cache_G --norm cache \
    --seeds 0 1 2 --channels weather --T 30 "${tgt[@]}" --bs 512 --lr 1e-4 --epochs 150 --patience 15 \
    --run-name "$rn" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
