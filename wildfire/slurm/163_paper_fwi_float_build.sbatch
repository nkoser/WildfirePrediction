#!/bin/bash
#SBATCH --job-name=wf_fwif
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Ungekappte FWI-Werte in Code-Einheiten (wildfire/paper/fwi_float.py), nur CPU, nur lesen am Wuerfel.
#   MODE=fit       Abbildungen BUI / FWI / ISI und Zuwachsmodelle DC / DMC, Pseudo-Kappung auf anderen Jahren
#                  -> derived_data/fwi_float/params${TAG}.json, Manifest p33_fwi_float_fit${TAG}
#   MODE=build     je Jahr (YEAR=... oder Array-Index i -> Jahr 2000 + i, z. B. --array=0-25)
#                  -> derived_data/fwi_float/uncapped_<jahr>${TAG}.npz, nodata_<jahr>${TAG}.npy,
#                     Manifest p33_fwi_float_build_<jahr>${TAG}
#   MODE=summary   NoData-Maske ueber alle Jahre, Jahrestabelle -> Manifest p33_fwi_float${TAG}
# Schreibt nur in das NEUE Verzeichnis derived_data/fwi_float/ und bricht ab, wenn eine Datei schon existiert.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
year=${YEAR:-$((2000 + ${SLURM_ARRAY_TASK_ID:-23}))}
"$PY" -m wildfire.paper.fwi_float --config "$CONFIG" --mode "${MODE:?MODE=fit|build|summary}" --year "$year" \
    --tag "${TAG:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
