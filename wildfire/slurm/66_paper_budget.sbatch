#!/bin/bash
#SBATCH --job-name=wf_budget
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Gleiches Trainingsbudget (20 Epochen) fuer den Vergleich mit Contagion v2 + LH-Start:
#   0-2  Logistic Hazard, 10 weitere Epochen ab eigenem Checkpoint -> ..._d128_lh20_s<seed>
#   3-5  Contagion v2 von Grund auf, 20 Epochen                    -> ..._d128_cg2long_s<seed>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); CK=/data/Wildfire/checkpoints/paper
if [ $i -lt 3 ]; then
  args=(--head lh --epochs 10 --init-from "$CK/sts_cache_G_next_cube_d128_lh_s{seed}_s{seed}.pt" --tag "_d128_lh20_s${seed}")
else
  args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --epochs 20 --tag "_d128_cg2long_s${seed}")
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --d 128 --depth 6 --steps-per-epoch 3000 --seeds $seed --save-ckpt "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
