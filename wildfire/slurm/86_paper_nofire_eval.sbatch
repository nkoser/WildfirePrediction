#!/bin/bash
#SBATCH --job-name=wf_nofire_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# U-Net im Paper-Protokoll mit und ohne Feuermaske nebeneinander (nofire_eval).
# Braucht 85 Index 0-1. Manifest p25_nofire_unet_cache_G.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.nofire_eval --config "$CONFIG" --cache cache_G \
    --runs _next_paperhp_w0 _next_paperhp_w0_nofire _next_paperhp_w0_nomodis ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
