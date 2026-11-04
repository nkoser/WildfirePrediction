#!/bin/bash
#SBATCH --job-name=wf_sts_spread_sfiou
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS: Teacher Forcing + Self-Forcing + weiches IoU der simulierten 13 x 13-Karte bis Tag 1/7/15/30
# (wildfire/paper/sts_spread_sfiou.py), Ellipse 11 x 11, Ring, von Grund auf. Je Variante 3 Seeds (seed = i % 3, v = i / 3):
#   v0 cg_sfiou05   TF 1.0, SF 1.0, IoU 0.5
#   v1 cg_sfiou20   TF 1.0, SF 1.0, IoU 2.0
# Ausgaben NEU: checkpoints/paper/sts_spread_sfiou_<name>_s<seed>_s<seed>.pt, Manifest p67_.
#   SMOKE=1: 1 Epoche x 20 Schritte, 2000 Trainings-/500 Auswertestichtage, Tag _smoke_<name>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) n=cg_sfiou05; args=(--spread ell --cg-kernel 11 --tf-weight 1.0 --sf-weight 1.0 --iou-weight 0.5 --mf-weight 0 --big-share 0) ;;
  1) n=cg_sfiou20; args=(--spread ell --cg-kernel 11 --tf-weight 1.0 --sf-weight 1.0 --iou-weight 2.0 --mf-weight 0 --big-share 0) ;;
esac
small=()
if [ -n "${SMOKE:-}" ]; then
  small=(--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500); n=smoke_$n
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_spread_sfiou --config "$CONFIG" --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" "${small[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
