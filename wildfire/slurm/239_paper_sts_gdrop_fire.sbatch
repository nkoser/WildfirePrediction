#!/bin/bash
#SBATCH --job-name=wf_sx_gdfire
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion mit Gruppen-Dropout und staerkerem Feuer-Dropout (wildfire/paper/sts_groupdrop_fire.py), sonst wie 217:
#   0-2   cg_gdrop_fd50   p_fire 0.50 (Feuer sichtbar in 25 %),   Seeds 0..2
#   3-5   cg_gdrop_fd75   p_fire 0.75 (Feuer sichtbar in 12.5 %), Seeds 0..2
# Ausgaben NEU: checkpoints/paper/sts_extreme_<name><SFX>_s<seed>_s<seed>.pt, p35_sts_extreme_..., p53_sts_groupdrop_fire_...
# Danach 240 (Shapley-Scores) und 241 (Aggregation).
# Rauchtest: EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500" SFX=_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
pf=(0.50 0.75); nm=(cg_gdrop_fd50 cg_gdrop_fd75)
n=${nm[$v]}${SFX:-}
if [[ -e "$EXP/data_manifests/p53_sts_groupdrop_fire_${n}_s${seed}.json" ]]; then
  echo "ABBRUCH: Manifest p53_sts_groupdrop_fire_${n}_s${seed}.json existiert schon"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_groupdrop_fire --config "$CONFIG" --p-fire "${pf[$v]}" \
    --seeds $seed --tag "_${n}_s${seed}" --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
