#!/bin/bash
#SBATCH --job-name=wf_hyb_sum
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Zusammenfassungen der Hybride zusammen mit den vorhandenen kalibrierten Laeufen (Originale unveraendert):
#   0  Formen      joint_eval summary: hyb_mc gegen cgs_mc und STS-LH (ind, scaled, mask)  -> summary_shapes_hyb.json
#   1  bedingt     joint_refc summary: hyb_cond / hyb0_cond gegen cgs_cond, lh_refc, lh_adj, emp
#                                                                                  -> refc_summary_hyb.json
#   2  Anzahlen    joint_counts summary mit Szenario-Scores                       -> counts_summary_hyb.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
case $i in
  0) "$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --tag _shapes_hyb --refs hyb_mc cgs_mc \
       --inputs shapes_cgs_calib.npz shapes_lh_calib.npz shapes_hyb_s0.npz shapes_hyb_s1.npz shapes_hyb_s2.npz ;;
  1) "$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part summary --tag _hyb \
       --refc-inputs refc_cgs.npz refc_lh.npz \
       --inputs cond_cgs.npz cond_lh.npz cond_hyb_s0.npz cond_hyb_s1.npz cond_hyb_s2.npz \
       --refs hyb_cond hyb0_cond cgs_cond lh_adj ;;
  2) "$PY" -m wildfire.paper.joint_counts --config "$CONFIG" --part summary --tag _hyb \
       --inputs counts_cgs.npz counts_lh.npz counts_hyb_s0.npz counts_hyb_s1.npz counts_hyb_s2.npz \
       --refs hyb_sim cgs_cg \
       --scen-inputs scenarios_cgs_calib.npz scenarios_lh_calib.npz scenarios_hyb_s0.npz scenarios_hyb_s1.npz \
                     scenarios_hyb_s2.npz --scen-refs hyb_sim cgs_cg ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
