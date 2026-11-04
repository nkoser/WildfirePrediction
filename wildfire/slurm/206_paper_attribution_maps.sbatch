#!/bin/bash
#SBATCH --job-name=wf_attr
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Integrated Gradients auf die Eingabe (wildfire/paper/attribution_maps.py): Pixel x Tag x Kanal, dt 1 und 30,
# FireMPC-Test-Split. Ein Modell je Array-Index: 0 lhb10  1 msv2  2 cg (von Grund auf)  3 cg2w (warm)
#   SMOKE=1 sbatch --array=0,1,2 206_...sbatch       80 Stichproben, 4 Schritte, 1 Seed -> attribution_smoke_<m>.npz
#   PLOT=1 sbatch --array= --gres= 206_...sbatch      Abbildungen aus allen vier (TAG=_smoke fuer den Rauchtest)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
C=/data/Wildfire/checkpoints
NAMES=(lhb10 msv2 cg cg2w)
M=("lhb10=$C/paper/sts_cache_G_next_cube_abl_lhbce10_s?_s?.pt"
   "msv2=$C/paper_msv2/sts_cache_G_next_cube_msv2_msv2_s?.pt"
   "cg=$C/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"
   "cg2w=$C/paper/sts_cache_G_next_cube_d128_cg2w_ema_nll_s?_s?.pt")
T=${TAG:-}
if [ -n "${PLOT:-}" ]; then
    tags=(); for n in "${NAMES[@]}"; do [ -f "$EXP/metrics/attribution_maps/attribution${T}_$n.npz" ] && tags+=("${T}_$n"); done
    "$PY" -m wildfire.paper.attribution_maps --config "$CONFIG" --plot-only --tags "${tags[@]}"
else
    i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ -n "${SMOKE:-}" ]; then T=${TAG:-_smoke3}; a=(--n 80 --steps 32 --n-refs 2 --max-seeds 1 --n-examples 4); else a=(${EXTRA:-}); fi
    "$PY" -m wildfire.paper.attribution_maps --config "$CONFIG" --models "${M[$i]}" --tag "${T}_${NAMES[$i]}" "${a[@]}"
fi
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
