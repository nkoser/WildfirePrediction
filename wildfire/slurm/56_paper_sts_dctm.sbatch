#!/bin/bash
#SBATCH --job-name=wf_paper_sts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# STS-DCTM: raeumlich-zeitlicher Transformer mit DCTM-Koepfen je Pixel und fuer
# den Ausschnitt. Vortraining auf Wuerfel-Stichtagen (51_paper_pixel_landmarks),
# Feintuning auf dem Paper-Cache (+ 55_paper_extra). Braucht 40_paper_prepare (_L61)
# und fuer --target next_patch 54_paper_next_labels.
#   EXTRA="--train-source cube --epochs 8 --steps-per-epoch 3000 --save-ckpt"
#   EXTRA="--init-from /data/Wildfire/checkpoints/paper/sts_..._s{seed}.pt --lr 1e-4"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
