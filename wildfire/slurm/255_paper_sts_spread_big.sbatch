#!/bin/bash
#SBATCH --job-name=wf_sts_spread_big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS, Ellipse 11 x 11, Ring + Teacher Forcing, aktive Braende im Training uebergewichtet
# (wildfire/paper/sts_spread_big.py; Daten aus 250). Je Variante 3 Seeds (seed = i % 3, v = i / 3), Tag _<name>_s<seed>:
#   v0 cg_bigtf10   50 % aktive Stichtage je Batch, TF lam 1.0   (Vergleich: 251 cg_ell11ringtf10)
#   v1 cg_bigtf30   50 % aktive Stichtage je Batch, TF lam 3.0
# aktiv = >= 5 Feuerpixel (23 x 23) an den letzten 3 Eingabetagen.
# Ausgaben NEU: checkpoints/paper/sts_spread_big_<name>_s<seed>_s<seed>.pt, derived_data/calib_cache_ext/,
# Manifest p62_sts_spread_big_<name>_s<seed>.
#   SMOKE=1: 200 Schritte x 1 Epoche, 3000 Trainings-/Auswertestichtage, Tag _smoke_<name>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) n=cg_bigtf10; args=(--spread ell --cg-kernel 11 --tf-weight 1.0 --big-thr 5 --big-days 3 --big-share 0.5) ;;
  1) n=cg_bigtf30; args=(--spread ell --cg-kernel 11 --tf-weight 3.0 --big-thr 5 --big-days 3 --big-share 0.5) ;;
esac
small=()
if [ -n "${SMOKE:-}" ]; then
  small=(--epochs 1 --steps-per-epoch 200 --limit-train 3000 --limit-eval 3000); n=smoke_$n
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_spread_big --config "$CONFIG" --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" "${small[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
