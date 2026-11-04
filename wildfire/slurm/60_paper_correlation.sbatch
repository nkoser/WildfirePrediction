#!/bin/bash
#SBATCH --job-name=wf_paper_corr
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Korrelationsanalyse der 55 Eingabekanaele (wildfire/paper/correlation.py):
# Spearman roh / im Fenster / je Jahreszeit, Lags, Cluster, Teilkorrelationen,
# VIF, Rechenbeziehungen, Trennschaerfe gegen Feuer an D+1. Ohne GPU.
# Braucht 40_paper_prepare (_L61) und 54 fuer D+1.
#   EXTRA="--reduce mean"   EXTRA="--split all"   EXTRA="--max-samples 5000"
# Ergebnis: $EXP/reports/correlation_<tag>/report.md, Abbildungen unter
# $EXP/figures/correlation_<tag>/, Manifest p19_correlation_<tag>.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.correlation --config "$CONFIG" --cache "${CACHE:-cache_G}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
