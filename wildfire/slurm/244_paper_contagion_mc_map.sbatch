#!/bin/bash
#SBATCH --job-name=wf_cg_mc_map
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Contagion-STS (abl_cg_scratch, 3 Seeds) als Simulation ueber die ganze Teaser-Region (contagion_mc_map.py):
# globales Mean-Field und Monte-Carlo-Pfade statt Ausbreitung nur im 13 x 13-Fenster.
#   -> figures/teaser_mc/ (Panels), metrics/teaser_mc/ (Kennzahlen, Pixel b). Braucht 243 (teaser_oracle/summary.json).
# Bricht ab, wenn eine Ausgabe schon existiert (nichts ueberschreiben).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_mc_map --config "$CONFIG" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
