#!/bin/bash
#SBATCH --job-name=wf_contagion
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS (paper/contagion.py): Zuendungs-Hazard + gelernter
# Ausbreitungskern, ueber tau gerollt. Gleiches Backbone, Vortraining und
# Auswahl wie 60_paper_sts_heads; Index = Seed.
# Checkpoints: sts_cache_G_next_cube_d128_contagion_s<seed>_s<seed>.pt
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
seed=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --d 128 --depth 6 --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt \
    --head contagion --tag "_d128_contagion_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
