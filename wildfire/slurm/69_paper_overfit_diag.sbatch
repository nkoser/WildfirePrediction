#!/bin/bash
#SBATCH --job-name=wf_ofdiag
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Overfitting-Diagnose: Logistic Hazard mit EMA, 12 Epochen OHNE fruehen Abbruch.
# 10 % der raeumlichen 64 x 64-Bloecke des Wuerfels bleiben aus dem Training (13 px Puffer).
# Je Epoche Survival-Log-Likelihood auf
#   train    zufaellige Trainings-Stichtage                 (lernt das Modell weiter?)
#   holdout  zufaellige Stichtage an neuen Orten, 2000-2019  (verallgemeinert es raeumlich?)
#   valyears zufaellige Stichtage 2020-2022                 (verallgemeinert es auf neue Jahre?)
# plus die Val-PR-AUC im Paper-Protokoll (bisherige Auswahlmetrik).
#   0  d 128, 6 Bloecke  -> ..._d128_lh_diag_s0
#   1  d 64, 4 Bloecke   -> ..._d64_lh_diag_s0
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
case $i in
  0) args=(--d 128 --depth 6 --tag "_d128_lh_diag_s0") ;;
  1) args=(--d 64 --depth 4 --tag "_d64_lh_diag_s0") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --head lh --epochs 12 --patience 100 --steps-per-epoch 3000 --seeds 0 --ema 0.999 \
    --diag-holdout 0.1 "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
