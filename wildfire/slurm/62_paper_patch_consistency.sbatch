#!/bin/bash
#SBATCH --job-name=wf_patch_cons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Konsistenz von Pixel- und Ausschnittskopf (Frechet-Schranken) und Clayton-/
# Gamma-Frailty-Aggregation der Pixel-Survivals ohne neues Training, theta auf
# den Val-Stichtagen. Braucht 57_paper_surv_eval (Stichtage) und die STS-Checkpoints.
#   EXTRA="--models sts:/data/Wildfire/checkpoints/paper/sts_..._s*.pt --tag _x"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.patch_consistency --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
