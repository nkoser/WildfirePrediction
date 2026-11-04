#!/bin/bash
#SBATCH --job-name=wf_sx_table
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Zusammenfassung der sts_extreme-Laeufe (166/167): Seed-Mittel, Ensemble, O/E@30 je Testjahr, Paper-PR-AUC je dt
# (wildfire/paper/sts_extreme_table.py), nur CPU. Ausgabe NEU: metrics/sts_extreme${TAG}/,
# derived_data/calib_cache_ext/ens_<name>.npz, Manifest p36_sts_extreme_table${TAG}.
#   NAMES="lh_ref lh_reg" REFS="abl_lh" TAG=_lh sbatch 168_paper_sts_extreme_table.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.sts_extreme_table --config "$CONFIG" \
    --names ${NAMES:-lh_ref lh_reg lh_px lh_both lh_mono lh_all cg_ref cg_both cg_mono cg_all} \
    --refs ${REFS:-abl_lh abl_cg_scratch} --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
