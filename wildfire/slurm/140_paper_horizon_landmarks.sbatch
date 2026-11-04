#!/bin/bash
#SBATCH --job-name=wf_hz_lm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Horizontkonsistenz (Stapel getrennter Einzelklassifikatoren bce2/8/16/31 gegen F(tau) von STS-LH und Contagion-STS)
# und Testjahre 2023/2024/2025 mit Block-Bootstrap, auf den 100.000 zufaelligen Test-Stichtagen
# (paper/horizon_landmarks.py, ueber den Wrapper contagion_linear, damit lincg laedt).
#   sbatch 140_...sbatch          voller Lauf -> metrics/paper_horizon/landmarks.{json,npz}
#   sbatch 140_...sbatch smoke    Rauchtest   -> metrics/paper_horizon/landmarks_smoke.*
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${1:-}" = smoke ]; then
  a=(--max-test 4000 --max-val 2000 --boot 20 --tag _smoke)
else
  a=()
fi
"$PY" -m wildfire.paper.contagion_linear wildfire.paper.horizon_landmarks --config "$CONFIG" "${a[@]}" "${@:2}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
