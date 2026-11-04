#!/bin/bash
#SBATCH --job-name=wf_split_var
#SBATCH --array=0-77
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Varianten-Ablation (166 sts_extreme, 159 sts_focus, 148 cg_fdrop_mono) auf den Kandidaten-Splits, gleiches Protokoll
# (Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke, EMA 0.999, 10 x 3000 Schritte), aber Wuerfel-, Survival- und Paper-
# Testdaten des Splits: --cube-dir paper_pixel_<sp> --surv-dir surv_eval_<sp> --cache cache_G_<yearstrat|blockstrat>.
# seed = i % 3, Variante v = (i / 3) % 13, Split sp = i / 39 (0 ys = year_strat, 1 bs = block_strat):
#   0 lh_ref  1 lh_fdrop  2 lh_resid  3 lh_ignw  4 lh_fdropall  5 lh_fdrop_mono  6 lh_fdrop_mono_rf
#   7 cg_ref  8 cg_mono   9 cg_fdrop  10 cg_ignw  11 cg_fdrop_mono  12 cg_fdrop_mono_rf
# Tag _<sp>_<name>_s<seed>; Ausgaben NEU: checkpoints/paper/sts_extreme_<sp>_<name>_s<seed>_s<seed>.pt,
# calib_cache_ext/sts_extreme_<sp>_<name>_s<seed>*.npz, p35_/p44_..., sts_focus/coef_<sp>_lh_resid_s<seed>.npz.
# Braucht 233 (Stichtage) und 225/229 (Caches). Danach 235 (focus_eval), 236 (Tabelle sts_extreme), 237 (Uebersicht).
# Rauchtest: SFX=_smoke SP_SFX=_smoke EXTRA="--epochs 1 --steps-per-epoch 20 --limit-eval 500"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$(((i / 3) % 13)); s=$((i / 39))
sps=(ys bs); caches=(cache_G_yearstrat cache_G_blockstrat); sp=${sps[$s]}
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
mod=sts_focus
case $v in
  0)  n=lh_ref;           mod=sts_extreme; args=(--head lh --ctx none) ;;
  1)  n=lh_fdrop;         args=(--head lh --variants fdrop) ;;
  2)  n=lh_resid;         args=(--head lh --variants resid) ;;
  3)  n=lh_ignw;          args=(--head lh --variants ignw) ;;
  4)  n=lh_fdropall;      args=(--head lh --variants fdropall) ;;
  5)  n=lh_fdrop_mono;    args=(--head lh --variants fdrop --mono 1) ;;
  6)  n=lh_fdrop_mono_rf; args=(--head lh --variants fdrop --mono 1 --pad reflect) ;;
  7)  n=cg_ref;           mod=sts_extreme; args=("${cg[@]}" --ctx none) ;;
  8)  n=cg_mono;          mod=sts_extreme; args=("${cg[@]}" --ctx none --mono 1) ;;
  9)  n=cg_fdrop;         args=("${cg[@]}" --variants fdrop) ;;
  10) n=cg_ignw;          args=("${cg[@]}" --variants ignw) ;;
  11) n=cg_fdrop_mono;    args=("${cg[@]}" --variants fdrop --mono 1) ;;
  12) n=cg_fdrop_mono_rf; args=("${cg[@]}" --variants fdrop --mono 1 --pad reflect) ;;
esac
tag="_${sp}${SFX:-}_${n}_s${seed}"
if ls "$EXP/checkpoints/paper/sts_extreme${tag}_s"*.pt >/dev/null 2>&1; then echo "ABBRUCH: Checkpoint sts_extreme${tag} existiert schon"; exit 1; fi
echo "[split_var] $sp $n Seed $seed ($mod) -> Tag $tag"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.$mod --config "$CONFIG" --cache "${caches[$s]}" \
    --cube-dir "paper_pixel_${sp}${SP_SFX:-}" --surv-dir "surv_eval_${sp}${SP_SFX:-}" \
    --seeds $seed --tag "$tag" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
