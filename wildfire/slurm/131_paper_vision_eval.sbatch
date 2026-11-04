#!/bin/bash
#SBATCH --job-name=wf_vision_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Auswertung der Vision-Baselines (130) neben STS-LH (74 v0), alle ueber den Wrapper paper/vision_backbones.py:
#   surv_eval (100.000 zufaellige Test-Stichtage 2023-2025, --calibrate --subsets)  -> p16_surv_eval_vision.json
#   paper_metrics_ckpt (Paper-Protokoll, cache_G, Ziel D+1)                         -> p17_paper_metrics_cache_G_next_vision.json
# STS-LH laeuft im selben Aufruf mit (gleiche Stichtage, gleiche Kalibrierung), die Ensembles sind je 3 Seeds.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
W=(-m wildfire.paper.vision_backbones)
m=("sts_abl_utae:$CK/${G}_abl_utae_s?_s?.pt" "sts_abl_convlstm:$CK/${G}_abl_convlstm_s?_s?.pt"
   "sts_abl_lh:$CK/${G}_abl_lh_s?_s?.pt")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "${W[@]}" wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --calibrate --subsets \
    --models "${m[@]}" --tag _vision || exit 1
"$PY" "${W[@]}" wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
    --models "${m[@]}" --tag _vision
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
