#!/bin/bash
#SBATCH --job-name=wf_region_crn_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Kurztest fuer 98 MIT STS-LH (lh_ind + lh_patch), den 97 nicht abdeckt: 200 Stichtage, 4 Pfade,
# 50 Bootstrap-Ziehungen. Prueft den ganzen Weg bis zum Schreiben der JSON.
# Ausgabe NEU: metrics/paper_region/region_crn_smoke_lh.json (+ _nll.npz).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.region_eval_crn --config "$CONFIG" \
  --cg "$CK/sts_cache_G_next_cube_abl_cg_s?_s?.pt" --lh "$CK/sts_cache_G_next_cube_abl_lh_s?_s?.pt" \
  --n 200 --Ms 4 --boot 50 --tag _smoke_lh
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
