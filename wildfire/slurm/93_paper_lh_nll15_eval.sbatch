#!/bin/bash
#SBATCH --job-name=wf_lh_nll15_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Auswertung von 92 neben den 10-Epochen-Referenzen und Contagion von Grund auf:
# surv_eval (--calibrate --subsets), paper_metrics_ckpt (cache G, Ziel D+1) und
# calib_eval fuer das neue Modell. Manifeste mit Tag _nll15.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
models=()
for n in d64_lh_ema_nll d64_lh_ema_nll15 d64_cg2_ema_nll; do models+=("sts_${n}:$CK/sts_cache_G_next_cube_${n}_s?_s?.pt"); done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --calibrate --subsets \
    --models "${models[@]}" --tag "_nll15" || exit 1
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
    --models "${models[@]}" --tag "_nll15" || exit 1
"$PY" -m wildfire.paper.calib_eval --config "$CONFIG" --models "sts_lh_d64_ema_nll15:$CK/sts_cache_G_next_cube_d64_lh_ema_nll15_s?_s?.pt" \
    --tag "_sts_lh_d64_ema_nll15"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
