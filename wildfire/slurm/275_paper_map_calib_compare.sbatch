#!/bin/bash
#SBATCH --job-name=wf_map_calib
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Pixelkarten mehrerer Modelle, roh gegen kalibriert (curve, PIT, KS), an den drei Teaser-Ereignissen
# (map_calib_compare.py). Braucht die Karten aus 150 / 243 / 273 und die Caches aus 274.
# Ausgabe: metrics/map_calib, figures/map_calib (bricht ab, wenn vorhanden).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.map_calib_compare --config "$CONFIG" --models \
  "teaser_lh=STS-LH=sts_teasermap_lh_sts" \
  "teaser_cgs=Contagion-STS=sts_teasermap_cg_scratch_sts" \
  "teaser_dctm=STS + DCTM=sts_teasermap_dctm_sts" \
  "teaser_cox=STS + Cox=sts_teasermap_cox_sts" \
  "teaser_deephit=STS + DeepHit=sts_teasermap_deephit_sts" \
  "teaser_lho=STS-LH + weather oracle"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
