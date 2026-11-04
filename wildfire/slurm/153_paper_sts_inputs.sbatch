#!/bin/bash
#SBATCH --job-name=wf_sx_inputs
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Treiber- und Randdiagnose (wildfire/paper/sts_inputs.py), Protokoll wie 166/173 (sts_extreme):
#   0-2   lh_fireonly   LH, nur Feuermaske
#   3-5   cg_fireonly   Contagion ohne Warmstart, nur Feuermaske
#   6-8   lh_nofire     LH, Kanalsatz no_fire
#   9-11  cg_nofire     Contagion, no_fire (auch der Ausbreitungspfad leer)
#   12-14 lh_reflect    LH, alle Kanaele, Reflect-Padding im Stem
#   15-17 cg_reflect    Contagion, Reflect-Padding
#   18-20 lh_nomodis    LH, Kanalsatz no_modis (ohne jede MODIS-Beobachtung, auch ohne Thermalbaender)
#   21-23 cg_nomodis    Contagion, no_modis
#   24-26 lh_nothermal  LH, ohne Feuermaske, LST_day/night, BT_b20/21/31/32
#   27-29 cg_nothermal  Contagion, nothermal
# Ausgaben NEU: checkpoints/paper/sts_extreme_<name>_s<seed>_s<seed>.pt, calib_cache_ext/,
# p35_sts_extreme_<name>_s<seed>, p41_sts_inputs_<name>_s<seed>. Danach 168 mit NAMES=... und eigenem TAG.
# Rauchtest (SFX vor _s<seed>, damit die Checkpoint-Pruefung des vollen Laufs nicht anschlaegt): EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500" SFX=_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
case $v in
  0) n=lh_fireonly; args=(--head lh --inputs fireonly) ;;
  1) n=cg_fireonly; args=("${cg[@]}" --inputs fireonly) ;;
  2) n=lh_nofire;   args=(--head lh --inputs nofire) ;;
  3) n=cg_nofire;   args=("${cg[@]}" --inputs nofire) ;;
  4) n=lh_reflect;  args=(--head lh --pad reflect) ;;
  5) n=cg_reflect;  args=("${cg[@]}" --pad reflect) ;;
  6) n=lh_nomodis;   args=(--head lh --inputs nomodis) ;;
  7) n=cg_nomodis;   args=("${cg[@]}" --inputs nomodis) ;;
  8) n=lh_nothermal; args=(--head lh --inputs nothermal) ;;
  9) n=cg_nothermal; args=("${cg[@]}" --inputs nothermal) ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_inputs --config "$CONFIG" \
    --seeds $seed --tag "_${n}${SFX:-}_s${seed}" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
