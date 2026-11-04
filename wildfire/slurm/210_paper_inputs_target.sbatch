#!/bin/bash
#SBATCH --job-name=wf_inputs_target
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Eingabe-Ablation aus 153 (fireonly, nofire, nothermal, nomodis, Referenz; LH und Contagion) im
# Paper-Protokoll mit Ziel D (Cache-Label) und D+1 (wildfire/paper/sts_inputs_target.py). Nur Auswertung,
# gleiche Checkpoints; D+1 dient als Selbsttest gegen p35_sts_extreme_<name>.
# Ausgaben NEU: p46_sts_inputs_target${TAG}, metrics/sts_inputs_target${TAG}/summary.md.
#   Rauchtest: TAG=_smoke EXTRA="--models lh_ref:all --limit-eval 300" sbatch 210_paper_inputs_target.sbatch
#   Voll:      sbatch --dependency=afterok:5329 210_paper_inputs_target.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_inputs_target --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
