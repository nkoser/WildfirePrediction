#!/bin/bash
#SBATCH --job-name=wf_abl_extra
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Fehlende Zeilen der Ablation (Tab. 3), gleiches Protokoll wie 74 (Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke,
# EMA 0.999, Batch 32, lr 3e-4, 10 x 3000 Schritte, --select last), 3 Seeds; seed = i % 3, v = i / 3:
#   v0  local   Attention: Raum wie factorized, Zeit nur im lokalen 3 x 3-Umfeld (--attn local)
#   v1  pw0     nur Pixel-Likelihood (--patch-weight 0): Mitte vergleichbar, Ausschnittskopf untrainiert
#   v2  d128    Breite 128 statt 64
# Neue Tags _abl_{local,pw0,d128}_s<seed>; nichts Bestehendes wird ueberschrieben (sts_dctm bricht nicht ab,
# darum vorher pruefen, dass der Checkpoint fehlt).
#   SMOKE=1 sbatch --array=0 ...   Rauchtest: 1 Epoche x 50 Schritte, Tag _abl_local_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); CK=/data/Wildfire/checkpoints/paper
d=64
case $v in
  0) args=(--head lh --attn local); name=local ;;
  1) args=(--head lh --patch-weight 0); name=pw0 ;;
  2) args=(--head lh); d=128; name=d128 ;;
esac
tag="_abl_${name}_s${seed}"; ep=(--epochs 10 --steps-per-epoch 3000)
if [ -n "${SMOKE:-}" ]; then tag="_abl_${name}_smoke"; ep=(--epochs 1 --steps-per-epoch 50 --limit-eval 2000); fi
out="$CK/sts_cache_G_next_cube${tag}_s${seed}.pt"
if [ -e "$out" ]; then echo "ABBRUCH: $out existiert schon"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache cache_G --train-source cube \
    --cube-dir paper_pixel --surv-dir surv_eval_test \
    --target next_patch --d $d --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999 \
    "${ep[@]}" --select last --seeds $seed --save-ckpt "${args[@]}" --tag "$tag"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
