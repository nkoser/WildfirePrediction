#!/bin/bash
#SBATCH --job-name=wf_split_cgscr
#SBATCH --array=0-29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-Zeilen der Ablation (74 v22-v25, v27) von Grund auf auf den Kandidaten-Splits, exakt wie 83 v4-v8
# (Ablationsprotokoll d 64, 4 Bloecke, 10 x 3000 Schritte, letzte EMA-Gewichte, kein --init-from). Die warm
# gestarteten 74 v21-v25/v27 laufen auf ys/bs nicht (alles von Grund auf); v21 entspricht v26 cg_scratch (Job 5812/5813).
# seed = i % 3, Variante v = (i / 3) % 5, Split = i / 15 (0 ys, 1 bs):
#   v0 cg_k1_scratch  v1 cg_k3_scratch  v2 cg_obs7_scratch  v3 cg_geo_scratch  v4 cg_rate1_scratch (braucht 276 k1)
# Ausgaben NEU: checkpoints/paper/sts_<cache>_next_cube_abl_<name>_s<seed>_s<seed>.pt
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$(((i / 3) % 5)); s=$((i / 15))
sps=(ys bs); caches=(cache_G_yearstrat cache_G_blockstrat); sp=${sps[$s]}; cache=${caches[$s]}
abl=(--cache "$cache" --train-source cube --cube-dir "paper_pixel_${sp}" --surv-dir "surv_eval_${sp}"
     --target next_patch --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999
     --epochs 10 --steps-per-epoch 3000 --select last)
case $v in
  0) n=cg_k1_scratch;    a=(--head contagion --cg-kernel 1 --cg-obs-days 30 --cg-burn curve) ;;
  1) n=cg_k3_scratch;    a=(--head contagion --cg-kernel 3 --cg-obs-days 30 --cg-burn curve) ;;
  2) n=cg_obs7_scratch;  a=(--head contagion --cg-kernel 7 --cg-obs-days 7 --cg-burn curve) ;;
  3) n=cg_geo_scratch;   a=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn geometric) ;;
  4) n=cg_rate1_scratch; a=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --rate-weight 1) ;;
esac
out="/data/Wildfire/checkpoints/paper/sts_${cache}_next_cube_abl_${n}_s${seed}_s${seed}.pt"
if [ -e "$out" ]; then echo "ABBRUCH: $out existiert schon"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --seeds $seed --save-ckpt "${abl[@]}" "${a[@]}" \
    --tag "_abl_${n}_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
