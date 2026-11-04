#!/bin/bash
#SBATCH --job-name=wf_sts_ctrl
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# STS-LH / STS-Contagion mit Bekaempfungsstatus als Pixel-Zusatz (sts_extreme --ctx px --ctx-dir ctx_ctrl*, aus 254),
# sonst exakt das Protokoll der Referenzen lh_ref / cg_ref (166 bzw. 234: Wuerfel-Stichtage, Ziel D+1, d 64,
# 4 Bloecke, EMA 0.999, 10 x 3000 Schritte). Vergleich: sts_extreme_[<sp>_]{lh,cg}_ref_s<seed>_s<seed>.pt.
# seed = i % 3, Kopf h = (i / 3) % 2 (lh, cg), Split s = i / 6 (0 t, 1 ys, 2 bs).
# Tag _[<sp>_]ctrl_<lh|cg>_s<seed>; Ausgaben NEU: checkpoints/paper/sts_extreme<tag>_s<seed>.pt, calib_cache_ext/,
# Manifest p35_sts_extreme<tag>.
# Rauchtest: SFX=_smoke EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 3000 --limit-eval 500" sbatch --array=0 ...
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); h=$(((i / 3) % 2)); s=$((i / 6))
pre=("" "ys_" "bs_"); caches=(cache_G cache_G_yearstrat cache_G_blockstrat)
cubes=(paper_pixel paper_pixel_ys paper_pixel_bs); survs=(surv_eval_test surv_eval_ys surv_eval_bs)
ctxs=(ctx_ctrl ctx_ctrl_ys ctx_ctrl_bs)
if (( h == 0 )); then n=lh; args=(--head lh); else n=cg; args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve); fi
tag="_${pre[$s]}ctrl${SFX:-}_${n}_s${seed}"
if ls "$EXP/checkpoints/paper/sts_extreme${tag}_s"*.pt >/dev/null 2>&1; then echo "ABBRUCH: Checkpoint sts_extreme${tag} existiert schon"; exit 1; fi
echo "[sts_ctrl] ${caches[$s]} $n Seed $seed -> Tag $tag"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_extreme --config "$CONFIG" --cache "${caches[$s]}" --cube-dir "${cubes[$s]}" \
    --surv-dir "${survs[$s]}" --ctx px --ctx-dir "${ctxs[$s]}" --seeds $seed --tag "$tag" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
