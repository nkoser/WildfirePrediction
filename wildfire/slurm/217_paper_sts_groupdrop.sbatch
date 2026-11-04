#!/bin/bash
#SBATCH --job-name=wf_sx_gdrop
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# STS mit zufaelligem Gruppen-Dropout (wildfire/paper/sts_groupdrop.py) als Grundlage fuer Shapley je Gruppe (218),
# Protokoll wie 153/166 (Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke, 10 x 3000 Schritte, Contagion ohne Warmstart):
#   0-2   lh_gdrop   LH, Seeds 0..2
#   3-5   cg_gdrop   Contagion (Kern 7, 30 Beobachtungstage, Brenndauer-Kurve)
# Ausgaben NEU: checkpoints/paper/sts_extreme_<name><SFX>_s<seed>_s<seed>.pt, p35_sts_extreme_..., p49_sts_groupdrop_...
# Rauchtest: EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500" SFX=_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) n=lh_gdrop; args=(--head lh) ;;
  1) n=cg_gdrop; args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve) ;;
esac
n=${n}${SFX:-}
if [[ -e "$EXP/data_manifests/p49_sts_groupdrop_${n}_s${seed}.json" ]]; then
  echo "ABBRUCH: Manifest p49_sts_groupdrop_${n}_s${seed}.json existiert schon"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_groupdrop --config "$CONFIG" \
    --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
