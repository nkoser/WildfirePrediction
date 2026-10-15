#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_surv_prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Survival auf den Paper-Daten, Schritt 1: Landmarks je Paper-Stichprobe mit
# Zeit bis zum ersten Feuer im 13x13-Ausschnitt (bis 100 Tage), Fenster aus dem
# Wuerfel in Cache-Reihenfolge, Selbsttest gegen den Cache.
#   CACHE=cache_G (Standard)   EXTRA="--per-sample 8 2 2"
# Braucht Wuerfel und 40_paper_prepare (T = 30) fuer den Selbsttest.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.surv_prepare --config "$CONFIG" --cache "${CACHE:-cache_G}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
