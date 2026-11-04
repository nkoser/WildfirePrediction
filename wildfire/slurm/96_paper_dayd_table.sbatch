#!/bin/bash
#SBATCH --job-name=wf_dayd_table
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Tabelle Ziel D gegen D+1, je mit paper55 / no_fire / no_modis (dayd_table).
# U-Nets aus 42/85, STS aus 68/83/85/94. Fehlende Laeufe werden ausgelassen,
# also jederzeit neu startbar. Manifest p26_dayd_table_cache_G.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.dayd_table --config "$CONFIG" --cache cache_G ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
