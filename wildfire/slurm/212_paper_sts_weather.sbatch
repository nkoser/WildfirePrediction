#!/bin/bash
#SBATCH --job-name=wf_sx_weather
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Eingabe-Ablation wie 153 (sts_inputs), zusaetzlicher Satz weather (wildfire/paper/sts_inputs_weather.py):
#   0-2   lh_weather   LH, nur Wetter/Feuerwetter/VPD (25 Kanaele)
#   3-5   cg_weather   Contagion ohne Warmstart, weather (Ausbreitungspfad leer)
# Ausgaben NEU: checkpoints/paper/sts_extreme_<name>_s<seed>_s<seed>.pt, p35_sts_extreme_<name>_s<seed>,
# p41_sts_inputs_<name>_s<seed>. Danach 214 (Ziel D gegen D+1).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
case $v in
  0) n=lh_weather; args=(--head lh --inputs weather) ;;
  1) n=cg_weather; args=("${cg[@]}" --inputs weather) ;;
esac
if [[ -e "$EXP/data_manifests/p41_sts_inputs_${n}${SFX:-}_s${seed}.json" ]]; then
  echo "ABBRUCH: Manifest p41_sts_inputs_${n}${SFX:-}_s${seed}.json existiert schon"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_inputs_weather --config "$CONFIG" \
    --seeds $seed --tag "_${n}${SFX:-}_s${seed}" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
