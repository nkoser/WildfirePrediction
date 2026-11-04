#!/bin/bash
#SBATCH --job-name=wf_ea_main_eval
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung von 99 wie 82, gepaart:
#   0  flaechentreu (99): surv_eval auf surv_eval_test_ea, paper_metrics_ckpt auf cache_G_ea -> Tag _main_ea
#   1  Originalpixel (68 v5, 83 v1): dieselben Auswertungen auf der Teilmenge cache_G_ea      -> Tag _main_orig
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
if [ "$i" -eq 0 ]; then c=cache_G_ea; sd=surv_eval_test_ea; tag=_main_ea; sub=()
else                     c=cache_G;    sd=surv_eval_test;    tag=_main_orig; sub=(--subset cache_G_ea); fi
models=()
for n in d64_lh_ema_nll d64_cg2_ema_nll; do models+=("sts_${n}:${CK}/sts_${c}_next_cube_${n}_s?_s?.pt"); done
echo "[ea-main-eval] $c: ${models[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name "$sd" --calibrate --subsets --area-weight \
    --models "${models[@]}" --tag "$tag" || exit 1
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache "$c" "${sub[@]}" --models "${models[@]}" --tag "$tag"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
