#!/bin/bash
#SBATCH --job-name=wf_strat_sts
#SBATCH --array=0-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# STS-LH und STS-Contagion (K 7, 30 Beobachtungstage, Brennkurve) von null, NUR auf den Cache-Stichproben des
# Train-Splits (sts_dctm --train-source cache, wie Ablation v30 paperonly_prauc): die Wuerfel-Stichtage 2000-2019
# ueberlappen den stratifizierten Test (alle Jahre), Wuerfel-Training waere dort ein Leck. Damit der Split-Vergleich
# nicht mit der Trainingsquelle vermengt ist, laeuft dasselbe Protokoll auch auf dem zeitlichen cache_G.
# d 64, 4 Bloecke, EMA 0.999, Batch 32, lr 3e-4, bis 30 Epochen (4 Stichtage je Stichprobe), Auswahl Val-PR-AUC,
# Geduld 10. seed = i % 3, Kopf h = (i / 3) % 2, Ziel t = (i / 6) % 2, Cache c = i / 12:
#   0-11   cache_G_stratified   (h: lh, cg; t: cache = Ziel D, next_patch = Ziel D+1)
#   12-23  cache_G (zeitlich, Referenz mit demselben Protokoll)
# Ausgaben NEU: p15_sts_dctm_<cache>{,_next}_cache_splitcmp_<lh|cg>_s<seed>.json,
#   checkpoints/paper/sts_<cache>{,_next}_cache_splitcmp_<lh|cg>_s<seed>_s<seed>.pt. Braucht 220.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); h=$(((i / 3) % 2)); t=$(((i / 6) % 2)); c=$((i / 12))
caches=(cache_G_stratified cache_G); C=${caches[$c]}
if (( t == 0 )); then tgt=cache; tt=""; else tgt=next_patch; tt=_next; fi
if (( h == 0 )); then n=lh; hd=(--head lh); else n=cg; hd=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve); fi
tag="_splitcmp${SFX:-}_${n}_s${seed}"
for f in "$EXP/data_manifests/p15_sts_dctm_${C}${tt}_cache${tag}.json" "$EXP/checkpoints/paper/sts_${C}${tt}_cache${tag}_s${seed}.pt"; do
  if [[ -e "$f" ]]; then echo "ABBRUCH: $f existiert schon"; exit 1; fi
done
echo "[strat] $C Ziel $tgt Kopf $n Seed $seed -> Tag $tag"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "$C" --train-source cache --target "$tgt" \
    --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999 --epochs 30 --steps-per-epoch 0 \
    --select prauc --patience 10 --seeds $seed --save-ckpt --tag "$tag" "${hd[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
