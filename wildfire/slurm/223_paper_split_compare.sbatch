#!/bin/bash
#SBATCH --job-name=wf_split_cmp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Tabelle zeitlicher gegen stratifizierten Split (wildfire/paper/split_compare.py), nur aus Manifesten von 221/222
# und den vorhandenen U-Net-Laeufen auf cache_G. Ausgaben NEU: metrics/split_compare${TAG}/summary.md,
# Manifest p47_split_compare${TAG}.   sbatch --dependency=afterok:<221>:<222> 223_paper_split_compare.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.split_compare --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
