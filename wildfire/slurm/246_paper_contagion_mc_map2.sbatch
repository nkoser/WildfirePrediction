#!/bin/bash
#SBATCH --job-name=wf_cg_mc_map2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Regions-Simulation (globales Mean-Field, Monte Carlo) fuer Contagion von Grund auf (abl_cg_scratch) und die
# gerichteten Kerne aus 245 (contagion_mc_map2.py). Braucht 243 und 245.
#   -> figures/teaser_mc2/, metrics/teaser_mc2/. MODELS="name=glob;..." uebersteuert die Modelle.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CK=/data/Wildfire/checkpoints/paper
M=${MODELS:-"cg_scratch=$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt;cg_k11free=$CK/sts_spread_cg_k11free_s?_s?.pt;cg_ell7=$CK/sts_spread_cg_ell7_s?_s?.pt;cg_ell11=$CK/sts_spread_cg_ell11_s?_s?.pt;cg_ell11fw=$CK/sts_spread_cg_ell11fw_s?_s?.pt"}
"$PY" -m wildfire.paper.contagion_mc_map2 --config "$CONFIG" --models "$M" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
