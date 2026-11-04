#!/bin/bash
#SBATCH --job-name=wf_ea_heads_eval
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung des Kopfvergleichs (Folie 14) auf Originalpixeln und flaechentreu, gepaart:
#   0  flaechentreu (81): surv_eval auf surv_eval_test_ea, paper_metrics_ckpt auf cache_G_ea
#      (dazu gegen die Labels des Papers: patchorig / centreorig)        -> Tag _heads_ea
#   1  Originalpixel (56, 60, 65): surv_eval wie bisher, paper_metrics_ckpt auf derselben
#      Teilmenge wie 0 (--subset cache_G_ea)                              -> Tag _heads_orig
# surv_eval jeweils mit --calibrate --subsets --area-weight (je Pixel und je km^2).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
if [ "$i" -eq 0 ]; then c=cache_G_ea; sd=surv_eval_test_ea; tag=_heads_ea; sub=()
else                     c=cache_G;    sd=surv_eval_test;    tag=_heads_orig; sub=(--subset cache_G_ea); fi
models=()
for h in "" _lh _deephit _cox _dctm_cloglog _cg2w; do
  models+=("sts${h}:${CK}/sts_${c}_next_cube_d128${h}_s?_s?.pt")
done
echo "[ea-eval] $c: ${models[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name "$sd" --calibrate --subsets --area-weight \
    --models "${models[@]}" --tag "$tag" || exit 1
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache "$c" "${sub[@]}" --models "${models[@]}" --tag "$tag"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
