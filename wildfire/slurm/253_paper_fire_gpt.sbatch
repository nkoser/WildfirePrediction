#!/bin/bash
#SBATCH --job-name=wf_fire_gpt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Tages-Transformer: ein Token je Tag, autoregressive Feuerkarte des naechsten Tages (wildfire/paper/fire_gpt.py;
# Ziele aus 252 / future_maps.py). Je Variante 3 Seeds (seed = i % 3, v = i / 3), Tag _<name>_s<seed>:
#   v0 gpt_tf     Teacher Forcing
#   v1 gpt_ss05   Scheduled Sampling, 50 % eigene Vorhersagen als Eingabe
#   v2 gpt_free   freies Ausrollen mit Gradient (wie in der Auswertung)
# Ausgaben NEU: checkpoints/paper/fire_gpt_<name>_s<seed>_s<seed>.pt, predictions/fire_gpt_<name>_s<seed>_s<seed>.npz,
# Manifest p60_fire_gpt_<name>_s<seed>.
#   SMOKE=1: 200 Schritte x 1 Epoche, 3000 Trainings-/Auswertestichtage, Tag _smoke_<name>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) n=gpt_tf;   args=(--mode tf) ;;
  1) n=gpt_ss05; args=(--mode ss --ss-prob 0.5) ;;
  2) n=gpt_free; args=(--mode free) ;;
esac
small=()
if [ -n "${SMOKE:-}" ]; then
  small=(--epochs 1 --steps-per-epoch 200 --limit-train 3000 --limit-eval 3000); n=smoke_$n
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.fire_gpt --config "$CONFIG" --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" "${small[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
