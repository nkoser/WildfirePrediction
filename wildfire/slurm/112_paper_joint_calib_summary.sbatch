#!/bin/bash
#SBATCH --job-name=wf_calib_sum
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Zusammenfassungen mit gepaartem Bootstrap fuer die nachkalibrierten Auswertungen.
#   0  Formen (110 Aufgaben 0-2)      1  Szenarien (110 Aufgaben 3-5)      2  bedingte Vorhersage (111)
# Ausgabe NEU: metrics/paper_joint/summary_{shapes,scenarios}_calib.json, cond_summary_calib.json.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
case $i in
  0) "$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --tag _shapes_calib --refs cg_mc cgs_mc \
       --inputs shapes_cg_calib.npz shapes_cgs_calib.npz shapes_lh_calib.npz ;;
  1) "$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --tag _scenarios_calib --refs cg_cg cgs_cg \
       --inputs scenarios_cg_calib.npz scenarios_cgs_calib.npz scenarios_lh_calib.npz ;;
  2) "$PY" -m wildfire.paper.joint_cond --config "$CONFIG" --part summary --tag _calib --refs cg_cond cgs_cond \
       --inputs cond_cg.npz cond_cgs.npz cond_lh.npz ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
