#!/bin/bash
#SBATCH --job-name=wf_centre_table
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Tabelle Mittelzelle gegen Ausschnitt, Ziel D und D+1 (centre_table): U-Nets aus
# 99_paper_unet_centre neben _paperhp / _next_paperhp_w0, jeweils auf beiden Labels
# bewertet. Fehlende Laeufe werden ausgelassen. Manifest p27_centre_table_cache_G.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.centre_table --config "$CONFIG" --cache cache_G ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
