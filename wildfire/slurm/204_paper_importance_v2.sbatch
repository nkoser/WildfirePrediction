#!/bin/bash
#SBATCH --job-name=wf_imp_v2
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Permutation Importance der neuen Modelle (wildfire/paper/importance_v2.py), FireMPC-Test-Split,
# Teilmengen all / ign30 / ign183 (neue Zuendungen). Ein Checkpoint je Array-Index:
#   0-2 lhb10 (STS-LH beta 1)  3-5 lhf (LH + Zustandsmerkmale)  6-8 msv2  9-11 cg (Contagion, von Grund auf)
#   12-14 cg2w (Contagion, warm gestartet aus LH)
#   SMOKE=1 sbatch --array=0,6,9 204_...sbatch          1500 Test-Stichproben, dt 1 und 30 -> *_smoke.json
#   AGGREGATE=1 sbatch --array= --gres= 204_...sbatch    CSV + Abbildungen (TAG=_smoke fuer den Rauchtest)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
C=/data/Wildfire/checkpoints
M=("lhb10=$C/paper/sts_cache_G_next_cube_abl_lhbce10_s?_s?.pt"
   "lhf=$C/paper_msv2/sts_cache_G_next_cube_msv2_lhf_s?.pt"
   "msv2=$C/paper_msv2/sts_cache_G_next_cube_msv2_msv2_s?.pt"
   "cg=$C/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"
   "cg2w=$C/paper/sts_cache_G_next_cube_d128_cg2w_ema_nll_s?_s?.pt")
if [ -n "${AGGREGATE:-}" ]; then
    "$PY" -m wildfire.paper.importance_v2 --config "$CONFIG" --aggregate --tag "${TAG:-}"
else
    i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ -n "${SMOKE:-}" ]; then a=(--max-test 1500 --dts 1 30 --tag _smoke); else a=(--tag "${TAG:-}" ${EXTRA:-}); fi
    "$PY" -m wildfire.paper.importance_v2 --config "$CONFIG" --models "${M[@]}" --index "$i" "${a[@]}"
fi
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
