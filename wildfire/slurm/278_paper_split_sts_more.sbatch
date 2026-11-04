#!/bin/bash
#SBATCH --job-name=wf_split_sts
#SBATCH --array=0-89
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Weitere STS-Trainings auf den Kandidaten-Splits, je Modell unveraendert und Protokoll wie auf dem zeitlichen Split
# (Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke, EMA 0.999, 10 x 3000 Schritte), Daten wie 234:
# --cube-dir paper_pixel_<sp> --surv-dir surv_eval_<sp> --cache cache_G_<yearstrat|blockstrat>.
# seed = i % 3, Variante v = (i / 3) % 15, Split sp = i / 45 (0 ys, 1 bs):
#   Eingabe-Ablation (wie 153/212, sts_inputs_weather kennt alle Saetze):
#   0 lh_fireonly  1 cg_fireonly  2 lh_nofire  3 cg_nofire  4 lh_nomodis  5 cg_nomodis
#   6 lh_nothermal 7 cg_nothermal 8 lh_weather 9 cg_weather
#   Gruppen-Dropout fuer Shapley (wie 217/239): 10 lh_gdrop  11 cg_gdrop  12 cg_gdrop_fd50  13 cg_gdrop_fd75
#   Monotonie (wie 166 v4): 14 lh_mono
# Tag _<sp>_<name>_s<seed> wie 234: checkpoints/paper/sts_extreme_<sp>_<name>_s<seed>_s<seed>.pt, p35_/p41_/p49_/p53_...
# Rauchtest: SFX=_smoke EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$(((i / 3) % 15)); s=$((i / 45))
sps=(ys bs); caches=(cache_G_yearstrat cache_G_blockstrat); sp=${sps[$s]}
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
mod=sts_inputs_weather
case $v in
  0)  n=lh_fireonly;   args=(--head lh --inputs fireonly) ;;
  1)  n=cg_fireonly;   args=("${cg[@]}" --inputs fireonly) ;;
  2)  n=lh_nofire;     args=(--head lh --inputs nofire) ;;
  3)  n=cg_nofire;     args=("${cg[@]}" --inputs nofire) ;;
  4)  n=lh_nomodis;    args=(--head lh --inputs nomodis) ;;
  5)  n=cg_nomodis;    args=("${cg[@]}" --inputs nomodis) ;;
  6)  n=lh_nothermal;  args=(--head lh --inputs nothermal) ;;
  7)  n=cg_nothermal;  args=("${cg[@]}" --inputs nothermal) ;;
  8)  n=lh_weather;    args=(--head lh --inputs weather) ;;
  9)  n=cg_weather;    args=("${cg[@]}" --inputs weather) ;;
  10) n=lh_gdrop;      mod=sts_groupdrop; args=(--head lh) ;;
  11) n=cg_gdrop;      mod=sts_groupdrop; args=("${cg[@]}") ;;
  12) n=cg_gdrop_fd50; mod=sts_groupdrop_fire; args=("${cg[@]}" --p-fire 0.50) ;;
  13) n=cg_gdrop_fd75; mod=sts_groupdrop_fire; args=("${cg[@]}" --p-fire 0.75) ;;
  14) n=lh_mono;       mod=sts_extreme; args=(--head lh --ctx none --mono 1) ;;
esac
tag="_${sp}${SFX:-}_${n}_s${seed}"
if ls "$EXP/checkpoints/paper/sts_extreme${tag}_s"*.pt >/dev/null 2>&1; then echo "ABBRUCH: Checkpoint sts_extreme${tag} existiert schon"; exit 1; fi
echo "[split_sts] $sp $n Seed $seed ($mod) -> Tag $tag"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.$mod --config "$CONFIG" --cache "${caches[$s]}" \
    --cube-dir "paper_pixel_${sp}${SP_SFX:-}" --surv-dir "surv_eval_${sp}${SP_SFX:-}" \
    --seeds $seed --tag "$tag" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
