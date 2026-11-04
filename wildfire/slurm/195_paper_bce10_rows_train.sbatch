#!/bin/bash
#SBATCH --job-name=wf_bce10_rows
#SBATCH --array=0-11%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Horizont-BCE beta 1.0 an tau 1/7/14/30 (paper/horizon_bce_any.py) fuer die uebrigen Zeilen von Tab. 1, damit alle
# denselben Kopf wie _abl_lhbce10 (182 v1) haben. Protokoll wie 74 (Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke,
# Batch 32, lr 3e-4, EMA 0.999, letzte Gewichte). Index i: Variante v = i / 3, Seed = i % 3.
#   v0  Contagion-STS von Grund auf, 10 x 3000      (wie 74 v26)  -> ..._abl_cgbce10_scratch_s<seed>
#   v1  Contagion-STS warm aus _abl_lhbce10, 3 Ep.  (wie 74 v21)  -> ..._abl_cgbce10_s<seed>
#   v2  U-TAE + LH, 10 x 3000                       (wie 130)     -> ..._abl_utae_bce10_s<seed>
#   v3  ConvLSTM + LH, 10 x 3000                    (wie 130)     -> ..._abl_convlstm_bce10_s<seed>
#   SMOKE=1 sbatch --array=0,3,6,9 ...   1 Epoche x 50 Schritte, Tag ..._smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); CK=/data/Wildfire/checkpoints/paper
base=(--config "$CONFIG" --cache cache_G --train-source cube --cube-dir paper_pixel --surv-dir surv_eval_test
      --target next_patch --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999
      --epochs 10 --steps-per-epoch 3000 --select last --seeds $seed --save-ckpt)
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
warm=(--init-from "$CK/sts_cache_G_next_cube_abl_lhbce10_s{seed}_s{seed}.pt" --init-backbone-only --epochs 3)
case $v in
  0) name=cgbce10_scratch; args=("${cg[@]}") ;;
  1) name=cgbce10;         args=("${cg[@]}" "${warm[@]}") ;;
  2) name=utae_bce10;      args=(--backbone utae --head lh) ;;
  3) name=convlstm_bce10;  args=(--backbone convlstm --head lh) ;;
  *) echo "ABBRUCH: Index $i"; exit 1 ;;
esac
tag="_abl_${name}_s${seed}"; ep=()
if [ -n "${SMOKE:-}" ]; then tag="_abl_${name}_smoke"; ep=(--epochs 1 --steps-per-epoch 50 --limit-eval 2000); fi
out="$CK/sts_cache_G_next_cube${tag}_s${seed}.pt"
if [ -e "$out" ]; then echo "ABBRUCH: $out existiert schon"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.horizon_bce_any train --hz-beta 1.0 --hz-taus 1 7 14 30 \
    "${base[@]}" "${args[@]}" "${ep[@]}" --tag "$tag"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
