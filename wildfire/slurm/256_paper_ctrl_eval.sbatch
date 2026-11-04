#!/bin/bash
#SBATCH --job-name=wf_ctrl_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Bekaempfungsstatus als Eingabe: sts_extreme --ctx px (255) gegen die Referenzen *_ref, Paper-Metriken je dt und
# Kalibrierung nach Status (wildfire/paper/ctrl_eval.py).
#   sbatch --dependency=afterok:<255-job> 256_paper_ctrl_eval.sbatch
#   Rauchtest: TAG=_smoke EXTRA="--variant ctrl_smoke --seeds 0 --splits t --limit 500" sbatch 256_...
# Ausgaben NEU: metrics/ctrl_eval${TAG}/{summary.md,metrics.csv,groups.csv,scores.npz}, Manifest p55_ctrl_eval${TAG}
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.ctrl_eval --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
