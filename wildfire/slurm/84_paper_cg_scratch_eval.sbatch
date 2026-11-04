#!/bin/bash
#SBATCH --job-name=wf_cg_scratch_eval
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung von 83 (Contagion-STS von Grund auf), je Gruppe warm gestartet gegen
# von Grund auf, mit der passenden Logistic-Hazard-Referenz: surv_eval (100.000
# zufaellige Stichtage 2023-2025, --calibrate --subsets) und paper_metrics_ckpt
# (Paper-Protokoll, cache G, Ziel D+1). Manifeste p16_surv_eval_scr<g>, p17_..._scr<g>.
#   0  d 128, EMA (68 v0/v1/v7/v9, 83 v0/v2)
#   1  d 64, EMA (68 v2/v3/v4/v5/v8, 83 v1)
#   2  B 256 (71 v1/v8, 83 v3)
#   3  Ablation d 64 (74 v0/v26, 83 v4-v8)
#   4  flaechentreu d 128 (81 v1/v5, 83 v9), dazu --area-weight
#   5  Ablation flaechentreu (74 mit CACHE=cache_G_ea v0/v26, 83 v10-v14), dazu --area-weight
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
g=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
c=cache_G; sd=surv_eval_test; aw=(); rec=(--rec-run _next_paperhp_w0)
case $g in
  0) names="d128_lh_ema d128_cg2w_ema d128_cg2_ema d128_lh_ema_nll d128_cg2w_ema_nll d128_cg2_ema_nll" ;;
  1) names="d64_lh_ema d64_cg2w_ema d64_cg2_ema d64_lh_ema_nll d64_cg2w_ema_nll d64_cg2_ema_nll" ;;
  2) names="d64_lh_bs256 d64_cg2w_bs256 d64_cg2_bs256" ;;
  3) names="abl_lh abl_cg abl_cg_scratch abl_cg_k1_scratch abl_cg_k3_scratch abl_cg_obs7_scratch abl_cg_geo_scratch abl_cg_rate1_scratch" ;;
  4) names="d128_lh d128_cg2w d128_cg2"; c=cache_G_ea; sd=surv_eval_test_ea; aw=(--area-weight); rec=() ;;
  5) names="abl_lh abl_cg abl_cg_scratch abl_cg_k1_scratch abl_cg_k3_scratch abl_cg_obs7_scratch abl_cg_geo_scratch abl_cg_rate1_scratch"
     c=cache_G_ea; sd=surv_eval_test_ea; aw=(--area-weight); rec=() ;;
esac
models=()
for n in $names; do models+=("sts_${n}:/data/Wildfire/checkpoints/paper/sts_${c}_next_cube_${n}_s?_s?.pt"); done
echo "[scr-eval] Gruppe $g ($c): ${models[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name "$sd" --calibrate --subsets "${aw[@]}" \
    --models "${models[@]}" --tag "_scr${g}" || exit 1
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache "$c" "${rec[@]}" \
    --models "${models[@]}" --tag "_scr${g}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
