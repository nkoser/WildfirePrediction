#!/bin/bash
#SBATCH --job-name=wf_refc_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Rauchtest fuer joint_refc (Neustart an Tag k, lh_adj, emp) und joint_counts (Kalibrierung der Zahl brennender
# Pixel, Szenarien je Jahr). Kalibrierung aus dem Rauchtest 108 (calib_smoke2), Zeilen wie cond_smoke2 (40 Stichtage).
# Ausgaben NEU: derived_data/joint_refc/smoke_*, metrics/paper_joint/{refc,counts}_smoke*, *_summary_smoke*.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
m=("cg=cg:$CK/${G}_abl_cg_s0_s0.pt" "lh=lh:$CK/${G}_abl_lh_s0_s0.pt")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e
"$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part build --build-name smoke --n 40
"$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part run --build-name smoke --n 40 --calib-dir calib_smoke2 \
  --M 2 --mc-rows 4096 --with-baselines --tag _smoke --models "${m[@]}"
"$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part summary --boot 50 --tag _smoke \
  --refc-inputs refc_smoke.npz --inputs cond_smoke2.npz --refs cg_cond lh_refc
"$PY" -m wildfire.paper.joint_counts --config "$CONFIG" --part run --n 64 --M 8 --bs 16 --calib-dir calib_smoke2 \
  --tag _smoke --models "${m[@]}"
"$PY" -m wildfire.paper.joint_counts --config "$CONFIG" --part summary --boot 50 --tag _smoke \
  --inputs counts_smoke.npz --refs cg_cg --scen-inputs scenarios_smoke2_calib.npz --scen-refs cg_cg
echo "[$(date -u +%FT%TZ)] exit 0"
