#!/bin/bash
#SBATCH --job-name=wf_lincg_joint
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Lineare Contagion, nachkalibriert (123 Aufgaben 1-3): dieselben gemeinsamen Auswertungen wie fuer cg/cgs/lh
# (110, 111, 117, 118), ueber den Wrapper paper/contagion_linear.py. Ausgaben NEU mit Name lincg:
#   0  Formen       metrics/paper_joint/shapes_lincg_calib.*        (wie 110 Aufgabe 0)
#   1  Szenarien    metrics/paper_joint/scenarios_lincg_calib.*     (wie 110 Aufgabe 3)
#   2  bedingt      metrics/paper_joint/cond_lincg.*                (wie 111)
#   3  Neustart     metrics/paper_joint/refc_lincg.*                (wie 117, Build test aus 116)
#   4  Anzahlen     metrics/paper_joint/counts_lincg.*              (wie 118)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
m="lincg=cg:/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_lincg_s?_s?.pt"
W=(-m wildfire.paper.contagion_linear)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $i in
  0) "$PY" "${W[@]}" wildfire.paper.joint_calib --config "$CONFIG" --calib-dir calib --M 32 --part shapes \
       --tag _lincg_calib --models "$m" ;;
  1) "$PY" "${W[@]}" wildfire.paper.joint_calib --config "$CONFIG" --calib-dir calib --M 32 --part scenarios \
       --tag _lincg_calib --models "$m" ;;
  2) "$PY" "${W[@]}" wildfire.paper.joint_cond --config "$CONFIG" --part run --calib-dir calib --tag _lincg \
       --models "$m" ;;
  3) "$PY" "${W[@]}" wildfire.paper.joint_refc --config "$CONFIG" --part run --build-name test --calib-dir calib \
       --tag _lincg --models "$m" ;;
  4) "$PY" "${W[@]}" wildfire.paper.joint_counts --config "$CONFIG" --part run --calib-dir calib --tag _lincg \
       --models "$m" ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
