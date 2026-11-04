#!/bin/bash
#SBATCH --job-name=wf_classic
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Klassische Vergleichsverfahren als zeitdiskrete Hazard-Modelle (paper/classic_surv.py): FWI, FWI-Gruppe,
# logistische Regression, HistGradientBoosting. Anpassung auf den zufaelligen Wuerfel-Stichtagen 2000-2019,
# Auswertung auf den zufaelligen Test-Stichtagen (surv_eval) und im Paper-Protokoll (cache_G, next_patch). Nur CPU.
#   sbatch 120_...sbatch            voller Lauf          -> metrics/paper_classic/{surv_eval,paper_metrics}.*
#   sbatch 120_...sbatch smoke      Rauchtest (klein)    -> metrics/paper_classic/*_smoke.*
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
if [ "${1:-}" = smoke ]; then
  a=(--max-train 20000 --max-test 3000 --gbm-iter 30 --tag _smoke)
else
  a=()
fi
"$PY" -m wildfire.paper.classic_surv --config "$CONFIG" "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
