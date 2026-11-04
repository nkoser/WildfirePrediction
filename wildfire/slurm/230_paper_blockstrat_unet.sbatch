#!/bin/bash
#SBATCH --job-name=wf_bs_unet
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# 3D-U-Net-Nachbau auf dem block-stratifizierten Split block_strat (cache_G_blockstrat), Paper-Hyperparameter wie p2_reconstruct_cache_G_cache_paperhp /
# _next_paperhp_w0 (paper55, T 30, bs 512, lr 1e-4, 150 Epochen, Geduld 15, Auswahl Val-BCE), 3 Seeds je dt:
#   0  Ziel D    (end 2 8 16 31)  -> p2_reconstruct_cache_G_blockstrat_cache_paperhp
#   1  Ziel D+1  (end 1 7 15 30)  -> p2_reconstruct_cache_G_blockstrat_cache_next_paperhp_w0
# Normierung: norm_stats.npz von cache_G (Symlink im Overlay, Config audit_splitsv2.yaml).
# Braucht 229. Ausgaben NEU (Manifest + predictions/paper/rec_cache_G_blockstrat_cache<run-name>.npz).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
C=cache_G_blockstrat
if (( i == 0 )); then
  tgt=(--end 2 8 16 31); rn="_paperhp${SFX:-}"
else
  tgt=(--target next_patch --end 1 7 15 30); rn="_next_paperhp_w0${SFX:-}"
fi
if [[ -e "$EXP/data_manifests/p2_reconstruct_${C}_cache${rn}.json" ]]; then
  echo "ABBRUCH: Manifest p2_reconstruct_${C}_cache${rn}.json existiert schon"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.reconstruct --config "$CONFIG" --cache "$C" --norm cache \
    --seeds 0 1 2 --channels paper55 --T 30 "${tgt[@]}" --bs 512 --lr 1e-4 --epochs 150 --patience 15 \
    --run-name "$rn" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
