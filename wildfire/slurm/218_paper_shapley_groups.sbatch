#!/bin/bash
#SBATCH --job-name=wf_shapley
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Exakte Shapley-Werte je Variablengruppe (wildfire/paper/shapley_groups.py) fuer die Netze aus 217:
#   Array 0..5 = Checkpoints in der Reihenfolge der --models-Liste (lh_gdrop s0..2, cg_gdrop s0..2), je 1024
#   Teilmengen x 4 dt; Scores nach metrics/shapley_groups${TAG}/scores_*.npz
#   AGGREGATE=1 sbatch --array=0 218_...sbatch   Shapley, Gegenprobe gegen sts_inputs-Retrains, summary.md, Bild
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ -n "${AGGREGATE:-}" ]; then
  "$PY" -m wildfire.paper.shapley_groups --config "$CONFIG" --tag "${TAG:-}" --aggregate ${EXTRA:-}
else
  "$PY" -m wildfire.paper.shapley_groups --config "$CONFIG" --tag "${TAG:-}" --index "${SLURM_ARRAY_TASK_ID:?}" ${EXTRA:-}
fi
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
