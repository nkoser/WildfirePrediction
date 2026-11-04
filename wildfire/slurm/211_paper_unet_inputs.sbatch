#!/bin/bash
#SBATCH --job-name=wf_unet_inputs
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# 3D-U-Net-Nachbau (Paper-Hyperparameter wie p2_reconstruct_cache_G_cache_paperhp / _next_paperhp_w0) mit den
# Kanalsaetzen der Eingabe-Ablation, die channels.py nicht kennt (wildfire/paper/reconstruct_inputs.py):
#   0  no_thermal  Ziel D    (end 2 8 16 31)   -> p2_reconstruct_cache_G_cache_paperhp_no_thermal
#   1  fire_only   Ziel D                      -> p2_reconstruct_cache_G_cache_paperhp_fire_only
#   2  no_thermal  Ziel D+1  (end 1 7 15 30)   -> p2_reconstruct_cache_G_cache_next_paperhp_w0_no_thermal
#   3  fire_only   Ziel D+1                    -> p2_reconstruct_cache_G_cache_next_paperhp_w0_fire_only
# 3 Seeds je dt. Ausgaben NEU (Manifest + predictions/paper/rec_cache_G_cache<run-name>.npz).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
sets=(no_thermal fire_only no_thermal fire_only)
cs=${sets[$i]}
if (( i < 2 )); then
  tgt=(--end 2 8 16 31); rn="_paperhp_${cs}${SFX:-}"
else
  tgt=(--target next_patch --end 1 7 15 30); rn="_next_paperhp_w0_${cs}${SFX:-}"
fi
if [[ -e "$EXP/data_manifests/p2_reconstruct_cache_G_cache${rn}.json" ]]; then
  echo "ABBRUCH: Manifest p2_reconstruct_cache_G_cache${rn}.json existiert schon"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.reconstruct_inputs --config "$CONFIG" --cache cache_G --norm cache \
    --seeds 0 1 2 --channels "$cs" --T 30 "${tgt[@]}" --bs 512 --lr 1e-4 --epochs 150 --patience 15 \
    --run-name "$rn" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
