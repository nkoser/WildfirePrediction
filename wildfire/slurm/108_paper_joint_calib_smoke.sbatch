#!/bin/bash
#SBATCH --job-name=wf_calib_smoke2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Rauchtest fuer joint_calib (Nachkalibrierung) und joint_cond (bedingte Vorhersage).
# Zuerst die Ereignisse je Pixel der Val-Stichtage (echte Datei, nur wenn sie fehlt; braucht 109 ohnehin),
# dann Fit mit 256 Val-Stichtagen, shapes / scenarios mit 64 Test-Stichtagen, cond mit 40 Stichtagen.
# Ausgaben NEU: derived_data/joint_calib/val_pixel_events.npz, metrics/paper_joint/calib_smoke2/,
#               metrics/paper_joint/*_smoke_calib*, cond_smoke*.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
CJ=/data/Wildfire/checkpoints/paper_joint
G=sts_cache_G_next_cube
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e
[ -f /data/Wildfire/derived_data/joint_calib/val_pixel_events.npz ] || \
  "$PY" -m wildfire.paper.joint_calib --config "$CONFIG" --part events
m=("cg=cg:$CK/${G}_abl_cg_s0_s0.pt" "lh=lh:$CK/${G}_abl_lh_s0_s0.pt" "lhrm_w=lh:$CJ/${G}_lhreg_mask_warm_s0.pt")
"$PY" -m wildfire.paper.joint_calib --config "$CONFIG" --part fit --calib-dir calib_smoke2 --n-val 256 --J 4 --M 2 \
  --rows 4096 --cg-line -2 0 --cg-step 0.5 --cg-min-step 0.5 --lh-step 0.5 --models "${m[@]}"
ev=(--calib-dir calib_smoke2 --n 64 --M 2 --bs 16)
"$PY" -m wildfire.paper.joint_calib --config "$CONFIG" --part shapes "${ev[@]}" --tag _smoke2_calib --models "${m[@]}"
"$PY" -m wildfire.paper.joint_calib --config "$CONFIG" --part scenarios "${ev[@]}" --tag _smoke2_calib \
  --models "${m[@]:0:2}"
"$PY" -m wildfire.paper.joint_cond --config "$CONFIG" --part run --calib-dir calib_smoke2 --n 40 --M 2 --rows 4096 \
  --tag _smoke2 --models "${m[@]:0:2}"
"$PY" -m wildfire.paper.joint_cond --config "$CONFIG" --part summary --boot 50 --tag _smoke2 \
  --inputs cond_smoke2.npz --refs cg_cond
echo "[$(date -u +%FT%TZ)] exit 0"
