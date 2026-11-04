#!/bin/bash
#SBATCH --job-name=wf_shap_fagg
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Aggregation (shapley_groups.py --aggregate, wie 218 mit AGGREGATE=1) je Variante aus 240, Gegenprobe nur gegen
# die cg_* Retrains (sts_inputs): 0 fd50, 1 fd75. Ausgaben NEU: metrics/shapley_groups_<fd>/summary.md, shapley.csv/png.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
fd=(fd50 fd75)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.shapley_groups --config "$CONFIG" --tag "_${fd[$i]}${SFX:-}" --aggregate --checks cg \
    --models "cg_gdrop=/data/Wildfire/checkpoints/paper/sts_extreme_cg_gdrop_${fd[$i]}${SFX:-}_s[0-9]_s[0-9].pt" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
