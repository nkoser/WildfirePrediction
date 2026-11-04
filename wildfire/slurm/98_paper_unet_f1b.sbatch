#!/bin/bash
#SBATCH --job-name=wf_unet_f1b
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Wie 97, aber mit einer Lesart der F1-Regel, die nicht in Epoche 0 haengen bleibt
# (97: Val-F1 bei 0,5 bleibt anfangs 0, jeder Lauf endete nach 16 Epochen mit Epoche 0).
# Index i: Lesart i / 6, Konfiguration i % 6 (wie 97: 0-2 Ziel D, 3-5 Ziel D+1;
# paper55 / no_fire / no_modis).
#   Lesart 0  F1 bei 0,5, Patience erst ab dem ersten F1 > 0 (--f1-wait) -> Laufname ..._f1w...
#   Lesart 1  bester F1 ueber alle Schwellen (--f1-mode max)            -> Laufname ..._f1max...
# Tabelle: 96_paper_dayd_table.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
v=$((i / 6)); k=$((i % 6))
cs=(paper55 no_fire no_modis paper55 no_fire no_modis)
if [ "$v" -eq 0 ]; then f1=(--f1-wait); s=f1w; else f1=(--f1-mode max); s=f1max; fi
if [ "$k" -lt 3 ]; then
  tg=cache; ends=(2 8 16 31); rn=("_paperhp_${s}" "_paperhp_${s}_no_fire" "_paperhp_${s}_no_modis")
else
  tg=next_patch; ends=(1 7 15 30); rn=("_next_paperhp_${s}" "_next_paperhp_${s}_nofire" "_next_paperhp_${s}_nomodis")
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.reconstruct_f1 --config "$CONFIG" --cache cache_G --norm cache --seeds 0 1 2 \
    --channels "${cs[$k]}" --target $tg --T 30 --end "${ends[@]}" --bs 512 --lr 1e-4 --epochs 150 \
    --patience 15 "${f1[@]}" --run-name "${rn[$((k % 3))]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
