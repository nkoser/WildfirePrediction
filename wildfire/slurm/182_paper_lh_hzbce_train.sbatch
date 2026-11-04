#!/bin/bash
#SBATCH --job-name=wf_hzbce
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# STS-LH mit Zusatzverlust beta * sum_tau BCE(F(tau)) an tau 1/7/14/30 (paper/horizon_bce.py), Protokoll wie 74 v0
# (_abl_lh): Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke, Batch 32, lr 3e-4, EMA 0.999, 10 x 3000 Schritte,
# letzte Gewichte. seed = i % 3, v = i / 3:  v0 beta 0.3 -> _abl_lhbce03_s<seed>,  v1 beta 1.0 -> _abl_lhbce10_s<seed>
#   SMOKE=1 sbatch --array=0 ...   1 Epoche x 50 Schritte, Tag _abl_lhbce03_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); CK=/data/Wildfire/checkpoints/paper
betas=(0.3 1.0); names=(lhbce03 lhbce10)
beta=${betas[$v]}; tag="_abl_${names[$v]}_s${seed}"; ep=(--epochs 10 --steps-per-epoch 3000)
if [ -n "${SMOKE:-}" ]; then tag="_abl_${names[$v]}_smoke"; ep=(--epochs 1 --steps-per-epoch 50 --limit-eval 2000); fi
out="$CK/sts_cache_G_next_cube${tag}_s${seed}.pt"
if [ -e "$out" ]; then echo "ABBRUCH: $out existiert schon"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.horizon_bce train --hz-beta "$beta" --hz-taus 1 7 14 30 \
    --config "$CONFIG" --cache cache_G --train-source cube --cube-dir paper_pixel --surv-dir surv_eval_test \
    --target next_patch --head lh --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999 \
    "${ep[@]}" --select last --seeds $seed --save-ckpt --tag "$tag"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
