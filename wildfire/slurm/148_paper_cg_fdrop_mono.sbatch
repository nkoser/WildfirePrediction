#!/bin/bash
#SBATCH --job-name=wf_cg_fdmono
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion ohne Warmstart mit Feuer-Dropout (p 0.5, nur FIRE_CH) und Monotonie (Gewicht 1 wie cg_mono),
# wildfire/paper/sts_focus.py, Protokoll wie 159:
#   0-2  cg_fdrop_mono      Zero-Padding
#   3-5  cg_fdrop_mono_rf   Reflect-Padding im Stem
# Ausgaben NEU: checkpoints/paper/sts_extreme_<name>_s<seed>_s<seed>.pt, calib_cache_ext/, p35_/p44_..._<name>_s<seed>.
# Rauchtest: SFX=_smoke EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3))
args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --variants fdrop --mono 1)
n=cg_fdrop_mono
if [ "$i" -ge 3 ]; then n=${n}_rf; args+=(--pad reflect); fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_focus --config "$CONFIG" \
    --seeds $seed --tag "_${n}${SFX:-}_s${seed}" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
