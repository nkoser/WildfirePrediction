#!/bin/bash
#SBATCH --job-name=wf_extreme
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Extremes Feuerwetter und beispiellose Bedingungen (wildfire/paper/extreme_eval.py), nur CPU:
# Feuerwetter-Klimatologie 2000-2019 an den Zellen der surv_eval-Stichtage aus dem Wuerfel, Test-
# Stichtage nach Perzentil-Baendern / ueber dem Zellmaximum, Nachkalibrierung waehrend der Saison.
# Liest die Hazards aus derived_data/calib_cache/ (Laeufe 70). Ausgabe NEU: Manifest
# p31_extreme_eval${TAG}, Klimatologie derived_data/extreme_eval/cell_clim_*_step${STEP}.npz.
#   STEP=30 TAG=_smoke CACHES="name:pfad"   Rauchtest (jeder 30. Saisontag, ein Modell)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
C=$EXP/derived_data/calib_cache
DEFAULT="lh_d128:$C/sts_lh_d128_ema_nll_sts_lh_d128_ema_nll.npz cg2w_d128:$C/sts_cg2w_d128_ema_nll_sts_cg2w_d128_ema_nll.npz
lh_d64:$C/sts_lh_d64_ema_nll_sts_lh_d64_ema_nll.npz cg2w_d64:$C/sts_cg2w_d64_ema_nll_sts_cg2w_d64_ema_nll.npz
cg2w:$C/sts_cg2w_ema_sts_cg2w_ema.npz"
read -ra CS <<< "$(echo ${CACHES:-$DEFAULT})"
"$PY" -m wildfire.paper.extreme_eval --config "$CONFIG" --caches "${CS[@]}" --day-step "${STEP:-1}" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
