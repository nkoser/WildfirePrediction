#!/bin/bash
#SBATCH --job-name=wf_ctrl11_ev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Auswertung Status-Versuch ab 2011 (wildfire/paper/ctrl11_eval.py): ctrl11 / zero11 (271) gegen ctrl (255) und *_ref.
# Ausgaben NEU: metrics/ctrl11_eval${TAG}/{summary.md,metrics.csv,groups.csv,scores.npz}, Manifest p55_ctrl11_eval${TAG}
# Rauchtest: TAG=_smoke EXTRA="--splits t --sfx _smoke --limit 2000 --boot 20" sbatch ...
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.ctrl11_eval --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
