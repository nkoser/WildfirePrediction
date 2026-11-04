#!/bin/bash
#SBATCH --job-name=wf_sfeat
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Zustandsmerkmale je Pixel fuer Multi-State v2 (wildfire/paper/state_features.py). Nur CPU.
# Index 0 train (+ train_types), 1 eval, 2 val, 3 paper -> derived_data/state_features/<teil>.npy
#   SMOKE=1 sbatch --array=0-3 200_...sbatch   je 2.000 Stichtage -> <teil>_smoke.npy
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
parts=(train eval val paper)
a=(--keys raw)   # Roh-MCD14A1: deckt sich exakt mit dem Feuerkanal (fire_index fehlen 2000-2019 Teile, Job 5131)
if [ -n "${SMOKE:-}" ]; then a+=(--max-n 2000 --tag "${STAG:-_smoke}"); fi
"$PY" -m wildfire.paper.state_features --config "$CONFIG" --part "${parts[$i]}" "${a[@]}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
