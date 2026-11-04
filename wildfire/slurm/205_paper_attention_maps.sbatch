#!/bin/bash
#SBATCH --job-name=wf_attn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Attention-Karten (wildfire/paper/attention_maps.py): raeumlich, zeitlich, Pooling des Ausschnittskopfs,
# FireMPC-Test-Split, dt 1 und 30, alle Seeds gemittelt; Abbildungen figures/attention_maps<tag>_*.png.
#   SMOKE=1 sbatch 205_...sbatch      300 Stichproben, ein Seed -> attention_smoke.npz
#   PLOT=1 sbatch --gres= 205_...sbatch   nur Abbildungen neu (TAG=...)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
C=/data/Wildfire/checkpoints
M=("lhb10=$C/paper/sts_cache_G_next_cube_abl_lhbce10_s?_s?.pt"
   "msv2=$C/paper_msv2/sts_cache_G_next_cube_msv2_msv2_s?.pt"
   "cg=$C/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"
   "cg2w=$C/paper/sts_cache_G_next_cube_d128_cg2w_ema_nll_s?_s?.pt")
if [ -n "${SMOKE:-}" ]; then a=(--max-n 300 --max-seeds 1 --tag _smoke)
elif [ -n "${PLOT:-}" ]; then a=(--plot-only --tag "${TAG:-}")
else a=(--tag "${TAG:-}" ${EXTRA:-}); fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
"$PY" -m wildfire.paper.attention_maps --config "$CONFIG" --models "${M[@]}" "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
