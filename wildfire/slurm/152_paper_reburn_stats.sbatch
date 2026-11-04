#!/bin/bash
#SBATCH --job-name=wf_reburn
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Wie oft brennt ein Pixel nach einer Pause wieder (wildfire/paper/reburn_stats.py)? Nur CPU, liest fire_index.npz.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.reburn_stats --config "$CONFIG"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
