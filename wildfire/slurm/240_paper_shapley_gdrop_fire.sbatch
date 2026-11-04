#!/bin/bash
#SBATCH --job-name=wf_shap_fire
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Exakte Shapley-Scores (wildfire/paper/shapley_groups.py, wie 218) fuer die Netze aus 239. Je Variante eigener
# Ausgabeordner; das Modell heisst dort cg_gdrop, damit die Gegenprobe (cg_* Retrains) passt:
#   0-2   fd50 Seeds 0..2 -> metrics/shapley_groups_fd50/scores_cg_gdrop_s<seed>.npz
#   3-5   fd75 Seeds 0..2 -> metrics/shapley_groups_fd75/
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
v=$((i / 3)); k=$((i % 3)); fd=(fd50 fd75)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.shapley_groups --config "$CONFIG" --tag "_${fd[$v]}${SFX:-}" --index "$k" --checks cg \
    --models "cg_gdrop=/data/Wildfire/checkpoints/paper/sts_extreme_cg_gdrop_${fd[$v]}${SFX:-}_s[0-9]_s[0-9].pt" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
