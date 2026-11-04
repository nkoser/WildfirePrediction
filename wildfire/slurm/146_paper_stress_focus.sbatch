#!/bin/bash
#SBATCH --job-name=wf_stress_focus
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Stresstest wie 160/169 (wildfire/paper/stress_test.py) fuer die besten sts_focus-Varianten; mit Huelle ueber
# wildfire/paper/focus_patch.py (resid braucht den Basis-Offset). Je Array-Index ein Checkpoint:
#   0-2  ${N1}  (Huelle ${W1:-none})     3-5  ${N2}  (Huelle ${W2:-none})
# Ausgabe NEU: derived_data/stress_test/<modell>_s<seed>${TAG}.npz (TAG Standard _focus); danach
#   MODELS="$N1 $N2" TAG=_focus sbatch --dependency=afterok:<id> 161_paper_stress_table.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
seed=$((i % 3))
if [ "$i" -lt 3 ]; then name=${N1:?}; w=${W1:-none}; else name=${N2:?}; w=${W2:-none}; fi
a=(--config "$CONFIG" --ckpt "$CK/sts_extreme_${name}_s${seed}_s${seed}.pt" --model-name "$name" --tag "${TAG:-_focus}" ${EXTRA:-})
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$w" = none ]; then
  "$PY" -m wildfire.paper.stress_test "${a[@]}"
else
  "$PY" -m wildfire.paper.focus_patch --target stress --wrap "$w" -- "${a[@]}"
fi
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
