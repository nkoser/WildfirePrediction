#!/bin/bash
#SBATCH --job-name=wf_joint_sum
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Zusammenfassung von 105 (paper/joint_eval.py --part summary): Mittel ueber Seeds je Stichtag und gepaarter
# Bootstrap (2.000 Ziehungen) gegen die Referenzen; negatives Mittel = Referenz besser.
#   0  Formen, trained       1  Szenarien, trained       2  Formen, default      3  Szenarien, default
# Braucht alle passenden Aufgaben von 105. Ausgabe NEU: metrics/paper_joint/summary_<...>.json.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
case $i in
  0) a=(--tag _shapes_trained --refs cg_mc cgs_mc cgtf_w_mc cgtf_s_mc --inputs shapes_cg_trained.npz
        shapes_cgs_trained.npz shapes_cgtf_w_trained.npz shapes_cgtf_s_trained.npz shapes_lh_trained.npz) ;;
  1) a=(--tag _scenarios_trained --refs cg_cg cgs_cg cgtf_w_cg cgtf_s_cg --inputs scenarios_cg_trained.npz
        scenarios_cgs_trained.npz scenarios_cgtf_w_trained.npz scenarios_cgtf_s_trained.npz scenarios_lh_trained.npz) ;;
  2) a=(--tag _shapes_default --refs cg_mc --inputs shapes_cg_default.npz shapes_lh_default.npz) ;;
  3) a=(--tag _scenarios_default --refs cg_cg --inputs scenarios_cg_default.npz scenarios_lh_default.npz) ;;
esac
"$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
