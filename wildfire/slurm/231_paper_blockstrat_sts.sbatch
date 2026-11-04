#!/bin/bash
#SBATCH --job-name=wf_bs_sts
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# STS-LH und STS-Contagion von null auf dem block-stratifizierten Split cache_G_blockstrat, exakt das Protokoll
# von 222 (nur Cache-Stichproben des Train-Splits, --train-source cache; Wuerfel-Stichtage 2000-2019 enthielten
# Test-Bloecke aller Jahre). Referenz auf dem zeitlichen cache_G kommt aus 222 (Aufgaben 12-23).
# seed = i % 3, Kopf h = (i / 3) % 2 (lh, cg), Ziel t = (i / 6) % 2 (cache = D, next_patch = D+1).
# Ausgaben NEU: p15_sts_dctm_cache_G_blockstrat{,_next}_cache_splitcmp_<lh|cg>_s<seed>.json und Checkpoints.
# Braucht 229.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); h=$(((i / 3) % 2)); t=$(((i / 6) % 2)); C=cache_G_blockstrat
if (( t == 0 )); then tgt=cache; tt=""; else tgt=next_patch; tt=_next; fi
if (( h == 0 )); then n=lh; hd=(--head lh); else n=cg; hd=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve); fi
tag="_splitcmp${SFX:-}_${n}_s${seed}"
for f in "$EXP/data_manifests/p15_sts_dctm_${C}${tt}_cache${tag}.json" "$EXP/checkpoints/paper/sts_${C}${tt}_cache${tag}_s${seed}.pt"; do
  if [[ -e "$f" ]]; then echo "ABBRUCH: $f existiert schon"; exit 1; fi
done
echo "[bs] $C Ziel $tgt Kopf $n Seed $seed -> Tag $tag"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "$C" --train-source cache --target "$tgt" \
    --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999 --epochs 30 --steps-per-epoch 0 \
    --select prauc --patience 10 --seeds $seed --save-ckpt --tag "$tag" "${hd[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
