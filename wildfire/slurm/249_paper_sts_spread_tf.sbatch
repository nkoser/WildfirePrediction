#!/bin/bash
#SBATCH --job-name=wf_sts_spread_tf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS, Ellipse 11 x 11 (wie 245 v2 cg_ell11) mit zusaetzlichem Teacher-Forcing-Verlust
# (wildfire/paper/sts_spread_tf.py, contagion_tf.py): NLL + lam * NLL aus der beobachteten Front.
# Je lam 3 Seeds (seed = i % 3, v = i / 3), Tag _<name>_s<seed>:
#   v0 cg_ell11tf05   lam = 0.5
#   v1 cg_ell11tf10   lam = 1.0
# Referenz: sts_spread_cg_ell11_s* (245), sts_extreme_cg_ref_s*. Ausgaben NEU: checkpoints/paper/
# sts_spread_tf_<name>_s<seed>_s<seed>.pt, derived_data/calib_cache_ext/, Manifest p56_sts_spread_tf_<name>_s<seed>.
#   SMOKE=1: 200 Schritte x 1 Epoche, 3000 Trainings-/Auswertestichtage, Tag _smoke_<name>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) n=cg_ell11tf05; args=(--spread ell --cg-kernel 11 --tf-weight 0.5) ;;
  1) n=cg_ell11tf10; args=(--spread ell --cg-kernel 11 --tf-weight 1.0) ;;
esac
small=()
if [ -n "${SMOKE:-}" ]; then
  small=(--epochs 1 --steps-per-epoch 200 --limit-train 3000 --limit-eval 3000); n=smoke_$n
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_spread_tf --config "$CONFIG" --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" "${small[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
