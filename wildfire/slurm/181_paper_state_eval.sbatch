#!/bin/bash
#SBATCH --job-name=wf_state_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Auswertung nach Zustand (U/O/B) und Typ des ersten Feuers (wildfire/paper/state_eval.py), Labels aus 180.
#   sbatch 181_...sbatch smoke     3.000 Test- / 2.000 Val-Stichtage, 20 Ziehungen -> state_eval_smoke.json
#   sbatch 181_...sbatch           alle Stichtage, 200 Ziehungen                    -> state_eval.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${1:-}" = smoke ]; then
  a=(--max-test 3000 --max-val 2000 --boot 20 --label-tag _smoke --tag _smoke)
else
  a=()
fi
"$PY" -m wildfire.paper.state_eval --config "$CONFIG" "${a[@]}" "${@:2}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
