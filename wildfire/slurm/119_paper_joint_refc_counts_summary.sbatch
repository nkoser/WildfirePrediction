#!/bin/bash
#SBATCH --job-name=wf_refc_sum
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
#   0  bedingte Vorhersage mit Neustart und Basismodellen (111 + 117), je Abstand und Testjahr
#   1  Zahl brennender Pixel (118) und Szenario-Scores je Testjahr (110 Aufgaben 3-5)
# Ausgabe NEU: metrics/paper_joint/refc_summary_calib.json, counts_summary_calib.json.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
case $i in
  0) "$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part summary --tag _calib \
       --refc-inputs refc_cg.npz refc_cgs.npz refc_lh.npz --inputs cond_cg.npz cond_cgs.npz cond_lh.npz \
       --refs cg_cond cgs_cond lh_refc lh_adj ;;
  1) "$PY" -m wildfire.paper.joint_counts --config "$CONFIG" --part summary --tag _calib \
       --inputs counts_cg.npz counts_cgs.npz counts_lh.npz --refs cg_cg cgs_cg \
       --scen-inputs scenarios_cg_calib.npz scenarios_cgs_calib.npz scenarios_lh_calib.npz --scen-refs cg_cg cgs_cg ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
