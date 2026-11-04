#!/bin/bash
#SBATCH --job-name=wf_t_scratch
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Zeitlicher Split: die Auswertungen, die bisher NUR mit warm gestarteten Modellen liefen, mit dem Gegenstueck von
# Grund auf wiederholen (alles von Grund auf). Die warmen Ergebnisse bleiben unveraendert liegen.
#   0  Risikokarten wie 71/72 (cg2w_ema) mit d128_cg2_ema (83 v0), gleiche Stichtage
#        -> figures/risk_maps/risk_cg2_ema_<datum>.{npz,png,pdf}, capture_stats_cg2_ema.json, p22_risk_map_cg2_ema
#   1  Extremwetter wie 115/116 mit den Scratch-Caches cg2_d128_ema_nll, cg2_d64_ema_nll, cg2_ema (statt cg2w_*)
#        -> p31_extreme_eval_scratch, metrics/extreme_eval/*_scratch.csv
#   2  Klimatologie-Blend wie 77 mit abl_cg_scratch (74 v26) statt abl_cg
#        -> p17_paper_metrics_cache_G_next_clim_scr, p25_clim_blend_scr
#   3-5 d128_lh20 wie 66 v0-v2, aber 20 Epochen von Grund auf statt 10 weitere ab d128_lh -> ..._d128_lh20_scratch_s<seed>
#   6  Auswertung des Budget-Vergleichs (nach 3-5): surv_eval (--calibrate --subsets) und paper_metrics_ckpt fuer
#        d128_lh, d128_lh20_scratch, d128_cg2long, d128_lh20 (warm, nur zum Vergleich) -> p16_/p17_..._budget_scr
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube
C=$EXP/derived_data/calib_cache
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $i in
  0)
    "$PY" -m wildfire.paper.risk_map --config "$CONFIG" --pattern "${CK}_d128_cg2_ema_s?_s?.pt" \
        --dates 2023-06-01 2023-08-10 2024-07-15 2025-05-28 --tag cg2_ema || exit 1
    "$PY" -m wildfire.paper.split_run --manifest-suffix _cg2_ema \
        --redirect risk_maps/capture_stats.json=risk_maps/capture_stats_cg2_ema.json \
        wildfire.paper.risk_map_plot --config "$CONFIG" --glob "/data/Wildfire/figures/risk_maps/risk_cg2_ema_*.npz" ;;
  1)
    cs=("lh_d128:$C/sts_lh_d128_ema_nll_sts_lh_d128_ema_nll.npz" "cg2_d128:$C/sts_cg2_d128_ema_nll_sts_cg2_d128_ema_nll.npz"
        "lh_d64:$C/sts_lh_d64_ema_nll_sts_lh_d64_ema_nll.npz" "cg2_d64:$C/sts_cg2_d64_ema_nll_sts_cg2_d64_ema_nll.npz"
        "cg2:$C/sts_cg2_ema_sts_cg2_ema.npz")
    "$PY" -m wildfire.paper.extreme_eval --config "$CONFIG" --caches "${cs[@]}" --day-step 1 --tag _scratch || exit 1
    "$PY" -m wildfire.paper.extreme_table --config "$CONFIG" --tag _scratch ;;
  2)
    models=("sts_abl_lh:${CK}_abl_lh_s?_s?.pt" "sts_abl_hist14:${CK}_abl_hist14_s?_s?.pt"
            "sts_abl_cg_scratch:${CK}_abl_cg_scratch_s?_s?.pt")
    "$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --models "${models[@]}" \
        --tag _clim_scr --save-scores || exit 1
    "$PY" -m wildfire.paper.clim_blend --config "$CONFIG" --cache cache_G --scores scores_cache_G_next_clim_scr.npz \
        --tag _scr ;;
  3|4|5)
    seed=$((i - 3)); out="${CK}_d128_lh20_scratch_s${seed}_s${seed}.pt"
    if [ -e "$out" ]; then echo "ABBRUCH: $out existiert schon"; exit 1; fi
    "$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache cache_G --train-source cube \
        --target next_patch --d 128 --depth 6 --steps-per-epoch 3000 --seeds $seed --save-ckpt \
        --head lh --epochs 20 --tag "_d128_lh20_scratch_s${seed}" ;;
  6)
    models=()
    for n in d128_lh d128_lh20_scratch d128_cg2long d128_lh20; do models+=("sts_${n}:${CK}_${n}_s?_s?.pt"); done
    "$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --calibrate --subsets \
        --models "${models[@]}" --tag _budget_scr || exit 1
    "$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
        --models "${models[@]}" --tag _budget_scr ;;
esac
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
