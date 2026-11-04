#!/bin/bash
#SBATCH --job-name=wf_cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Parameter, FLOPs und Inferenzzeit je Ausschnitt (paper/compute_cost.py) fuer STS-LH, Contagion-STS, lineare
# Contagion und das 3D-U-Net -> metrics/paper_horizon/compute_cost.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_linear wildfire.paper.compute_cost --config "$CONFIG" "$@"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
