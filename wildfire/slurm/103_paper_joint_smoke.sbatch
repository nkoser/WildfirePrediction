#!/bin/bash
#SBATCH --job-name=wf_joint_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Rauchtest fuer joint_train und joint_eval: jede Modellart mit 256 Stichtagen und 5 Schritten (warm und
# von Grund auf), dann shapes / scenarios auf 64 Test-Stichtagen mit 2 Pfaden, dann summary.
# Ausgaben NEU: checkpoints/paper_joint/*_smoke*, metrics/paper_joint/*_smoke*.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
CJ=/data/Wildfire/checkpoints/paper_joint
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
tr=(--limit 256 --epochs 1 --steps-per-epoch 5 --warmup 2)
set -e
"$PY" -m wildfire.paper.joint_train --config "$CONFIG" --model cg_tf "${tr[@]}" --tag _smoke
"$PY" -m wildfire.paper.joint_train --config "$CONFIG" --model cg_tf "${tr[@]}" --init-from= --tag _smokescr
"$PY" -m wildfire.paper.joint_train --config "$CONFIG" --model lh_region --region-mode fixed "${tr[@]}" --tag _smoke
"$PY" -m wildfire.paper.joint_train --config "$CONFIG" --model lh_region --region-mode mask "${tr[@]}" --tag _smoke
"$PY" -m wildfire.paper.joint_train --config "$CONFIG" --model lh_ign "${tr[@]}" --tag _smoke
ev=(--n 64 --M 2 --bs 16)
"$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part shapes "${ev[@]}" --tag _smoke --models \
  "cg=cg:$CK/sts_cache_G_next_cube_abl_cg_s0_s0.pt" "cgtf=cg:$CJ/sts_cache_G_next_cube_cgtf_smoke_s0.pt" \
  "lh=lh:$CK/sts_cache_G_next_cube_abl_lh_s0_s0.pt" "lhrf=lh:$CJ/sts_cache_G_next_cube_lhreg_fixed_smoke_s0.pt" \
  "lhrm=lh:$CJ/sts_cache_G_next_cube_lhreg_mask_smoke_s0.pt"
"$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part shapes "${ev[@]}" --bias default --tag _smoke_default \
  --models "cg=cg:$CK/sts_cache_G_next_cube_abl_cg_s0_s0.pt" "lh=lh:$CK/sts_cache_G_next_cube_abl_lh_s0_s0.pt"
"$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part scenarios "${ev[@]}" --tag _smoke --models \
  "cg=cg:$CK/sts_cache_G_next_cube_abl_cg_s0_s0.pt" "lh=lh:$CK/sts_cache_G_next_cube_abl_lh_s0_s0.pt"
"$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --boot 50 --tag _smoke \
  --inputs shapes_smoke.npz --refs cg_mc
"$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --boot 50 --tag _smoke_scen \
  --inputs scenarios_smoke.npz --refs cg_cg
"$PY" -m wildfire.paper.contagion_cause_eval --config "$CONFIG" --n 2000 --tag _smoke_lhign \
  --models "lh:$CJ/sts_cache_G_next_cube_lhign_smoke_s0.pt"
echo "[$(date -u +%FT%TZ)] exit 0"
