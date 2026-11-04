#!/bin/bash
#SBATCH --job-name=wf_lincg_sum
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Zusammenfassungen mit Bootstrap wie 112/119, zusaetzlich mit der linearen Contagion (124) als Eingabe und Referenz.
# Eigene Tags (*_lincg), die Zusammenfassungen aus 112/119 bleiben unberuehrt.
#   0 Formen   1 Szenarien   2 bedingt   3 Neustart   4 Anzahlen
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
cond=(cond_cg.npz cond_cgs.npz cond_lh.npz cond_lincg.npz)
scen=(scenarios_cg_calib.npz scenarios_cgs_calib.npz scenarios_lh_calib.npz scenarios_lincg_calib.npz)
case $i in
  0) "$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --tag _shapes_calib_lincg \
       --refs lincg_mc cg_mc --inputs shapes_cg_calib.npz shapes_cgs_calib.npz shapes_lh_calib.npz shapes_lincg_calib.npz ;;
  1) "$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --tag _scenarios_calib_lincg \
       --refs lincg_cg cg_cg --inputs "${scen[@]}" ;;
  2) "$PY" -m wildfire.paper.joint_cond --config "$CONFIG" --part summary --tag _calib_lincg \
       --refs lincg_cond cg_cond --inputs "${cond[@]}" ;;
  3) "$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part summary --tag _calib_lincg \
       --refc-inputs refc_cg.npz refc_cgs.npz refc_lh.npz refc_lincg.npz --inputs "${cond[@]}" \
       --refs lincg_cond cg_cond lh_refc lh_adj ;;
  4) "$PY" -m wildfire.paper.joint_counts --config "$CONFIG" --part summary --tag _calib_lincg \
       --inputs counts_cg.npz counts_cgs.npz counts_lh.npz counts_lincg.npz --refs lincg_cg cg_cg \
       --scen-inputs "${scen[@]}" --scen-refs lincg_cg cg_cg ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
