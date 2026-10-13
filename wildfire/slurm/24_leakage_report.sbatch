#!/bin/bash
#SBATCH --job-name=wf_leakage_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BEWUSST OHNE --partition: reiner CPU-Job.
# Rendert leakage_report.md deterministisch aus den Stufen-Manifesten.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.leakage_report --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
