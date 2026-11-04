#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_ea_build
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Flaechentreue Datensaetze (paper/ea_build.py, paper/equal_area.py): dieselben Stichtage,
# Ausschnitte aus 13 x 13 quadratischen Zellen von rund 5 x 5 km statt 13 x 13 Gitterpixeln.
#   0  train   paper_pixel      -> paper_pixel_ea       (Wuerfel-Stichtage fuer das Training)
#   1  surv    surv_eval_test   -> surv_eval_test_ea    (100.000 Test-, 50.000 Val-, KM-Stichtage)
#   2  cache   cache_G_L61      -> cache_G_ea_L61       (Paper-Stichproben, ok.npy = Fenster im Wuerfel)
#   3  rate    Tageslabels fuer paper_pixel_ea          (braucht 0)
#   4  future  Wetter-Oracle fuer paper_pixel_ea        (braucht 0)
# Erst 0-2 abschicken, dann 3-4 mit --dependency=afterok:<job>_0.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
STEPS=(train surv cache rate future)
step=${STEPS[${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}]}
"$PY" -m wildfire.paper.ea_build --config "$CONFIG" --step "$step" --workers "${SLURM_CPUS_PER_TASK:-8}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
