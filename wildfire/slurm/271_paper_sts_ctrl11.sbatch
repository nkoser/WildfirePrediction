#!/bin/bash
#SBATCH --job-name=wf_sts_c11
#SBATCH --array=0-35
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Wie 255 (Status als Pixel-Zusatz, Protokoll der Referenzen lh_ref / cg_ref), aber nur Wuerfel-Stichtage ab 2011
# (train_rows.npy aus 270). Variante v = i / 18: 0 ctrl11 (mit Status, ctx_ctrl11*), 1 zero11 (Status-Eingabe null,
# ctx_zero11* = Referenz auf derselben Teilmenge). Innerhalb: seed = j % 3, Kopf h = (j / 3) % 2 (lh, cg),
# Split s = j / 6 (0 t, 1 ys, 2 bs), j = i % 18.
# Tag _[<sp>_]<ctrl11|zero11>_<lh|cg>_s<seed>; Ausgaben NEU: checkpoints/paper/sts_extreme<tag>_s<seed>.pt,
# Manifest p35_sts_extreme<tag>.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
v=$((i / 18)); j=$((i % 18))
seed=$((j % 3)); h=$(((j / 3) % 2)); s=$((j / 6))
pre=("" "ys_" "bs_"); sfx=("" "_ys" "_bs"); caches=(cache_G cache_G_yearstrat cache_G_blockstrat)
cubes=(paper_pixel paper_pixel_ys paper_pixel_bs); survs=(surv_eval_test surv_eval_ys surv_eval_bs)
vars=(ctrl11 zero11)
ctxd="ctx_${vars[$v]}${sfx[$s]}"
if (( h == 0 )); then n=lh; args=(--head lh); else n=cg; args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve); fi
tag="_${pre[$s]}${vars[$v]}${SFX:-}_${n}_s${seed}"
if ls "$EXP/checkpoints/paper/sts_extreme${tag}_s"*.pt >/dev/null 2>&1; then echo "ABBRUCH: Checkpoint sts_extreme${tag} existiert schon"; exit 1; fi
echo "[sts_c11] ${caches[$s]} ${vars[$v]} $n Seed $seed ctx $ctxd -> Tag $tag"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_extreme --config "$CONFIG" --cache "${caches[$s]}" --cube-dir "${cubes[$s]}" \
    --surv-dir "${survs[$s]}" --ctx px --ctx-dir "$ctxd" --seeds $seed --tag "$tag" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
