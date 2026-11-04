#!/bin/bash
#SBATCH --job-name=wf_riskplot
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Teaser-Abbildungen und Trefferquoten aus den Karten von 71_paper_risk_map
# (wildfire/paper/risk_map_plot.py). Nur CPU.
#   EXTRA="--tau-main 7 --zoom-taus 1,7,30 --zoom lon0,lon1,lat0,lat1;..."  (ohne Leerzeichen)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.risk_map_plot --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
