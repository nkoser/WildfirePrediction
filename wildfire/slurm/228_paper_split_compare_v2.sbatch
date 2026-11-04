#!/bin/bash
#SBATCH --job-name=wf_split_cmp2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Tabelle zeitlich (cache_G) / stratifiziert (cache_G_stratified) / jahres-stratifiziert (cache_G_yearstrat),
# wildfire/paper/split_compare_v2.py, nur aus Manifesten von 221/222/226/227. Ausgaben NEU:
# metrics/split_compare_v2${TAG}/summary.md, Manifest p48_split_compare_v2${TAG}.
#   sbatch --dependency=afterok:<222>:<226>:<227> 228_paper_split_compare_v2.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.split_compare_v2 --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
