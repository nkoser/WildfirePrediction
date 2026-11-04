#!/bin/bash
#SBATCH --job-name=wf_classic_lit
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Verfahren aus der Feuer- und Survival-Literatur, an unser Protokoll angepasst (paper/classic_lit.py):
# Lasso-Logistik, GAM, Cox-PH, Weibull, IPCW-GBM je Horizont, XGBoost-AFT. Gleiche Daten und Auswertung wie 120.
#   sbatch 122_...sbatch            voller Lauf      -> metrics/paper_classic/lit_{surv_eval,paper_metrics}.*
#   sbatch 122_...sbatch smoke      Rauchtest        -> metrics/paper_classic/lit_*_smoke.*
#   weitere Argumente nach dem ersten gehen an classic_lit (z. B. smoke --tag _smoke2)
# xgboost ist nicht im Image: Versuch, es ohne Abhaengigkeiten in ein eigenes Verzeichnis zu legen und HINTEN an
# PYTHONPATH zu haengen (kann nichts im Image verdecken). Klappt das nicht, entfaellt nur xgb_aft. Nur CPU.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
DEPS=/data/Wildfire/pydeps_xgboost
if [ ! -d "$DEPS/xgboost" ]; then
  "$PY" -m pip install --no-deps --no-cache-dir --target "$DEPS" xgboost || echo "[lit] xgboost-Installation fehlgeschlagen"
fi
export PYTHONPATH="${PYTHONPATH}:$DEPS"
if [ "${1:-}" = smoke ]; then
  a=(--max-train 20000 --max-test 3000 --gbm-iter 30 --tag _smoke)
else
  a=()
fi
"$PY" -m wildfire.paper.classic_lit --config "$CONFIG" --threads "${SLURM_CPUS_PER_TASK:-16}" "${a[@]}" "${@:2}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
