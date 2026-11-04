#!/bin/bash
#SBATCH --job-name=wf_ctrl_ctx
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Bekaempfungsstatus (CWFIS Reported Fires, 252) als Pixel-Zusatzeingabe fuer sts_extreme --ctx px
# (wildfire/paper/ctrl_ctx.py). Aufgabe i: 0 t = cache_G (zeitlich), 1 ys = cache_G_yearstrat, 2 bs = cache_G_blockstrat.
# Ausgaben NEU: derived_data/ctx_ctrl{,_ys,_bs}/{train,surv_val,surv_eval}_px.npy, cache_px.npy, channels.json,
# Manifest p55_ctrl_ctx_ctx_ctrl{,_ys,_bs}. Danach 255 (Training).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
caches=(cache_G cache_G_yearstrat cache_G_blockstrat); cubes=(paper_pixel paper_pixel_ys paper_pixel_bs)
survs=(surv_eval_test surv_eval_ys surv_eval_bs); outs=(ctx_ctrl ctx_ctrl_ys ctx_ctrl_bs)
"$PY" -m wildfire.paper.ctrl_ctx --config "$CONFIG" --cache "${caches[$i]}" --cube-dir "${cubes[$i]}" \
    --surv-dir "${survs[$i]}" --out-name "${outs[$i]}${SFX:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
