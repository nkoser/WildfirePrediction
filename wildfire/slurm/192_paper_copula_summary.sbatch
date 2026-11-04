#!/bin/bash
#SBATCH --job-name=wf_cop_sum
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Zusammenfassungen der Frailty zusammen mit den vorhandenen kalibrierten Laeufen (STS-LH, Contagion-STS von Grund
# auf, Hawkes, Hybride v2); Originale unveraendert, eigene Tags *_copula:
#   0  Formen      joint_eval summary  -> summary_shapes_copula.json
#   1  bedingt     joint_refc summary  -> refc_summary_copula.json   (mit lh_refc, lh_adj, emp, hawkes_refc)
#   2  Anzahlen    joint_counts summary mit Szenario-Scores -> counts_summary_copula.json
#   3  Szenarien   joint_eval summary  -> summary_scenarios_copula.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
cop() { echo "${1}_copula_s0.npz ${1}_copula_s1.npz ${1}_copula_s2.npz"; }
hyb() { echo "${1}_hyb_s0.npz ${1}_hyb_s1.npz ${1}_hyb_s2.npz"; }
scen=(scenarios_cgs_calib.npz scenarios_lh_calib.npz scenarios_hawkes_calib.npz $(hyb scenarios) $(cop scenarios))
case $i in
  0) "$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --tag _shapes_copula \
       --refs cop_cf copg_cf --inputs shapes_cgs_calib.npz shapes_lh_calib.npz shapes_hawkes_calib.npz \
       $(hyb shapes) $(cop shapes) ;;
  1) "$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part summary --tag _copula \
       --refc-inputs refc_cgs.npz refc_lh.npz refc_hawkes.npz \
       --inputs cond_cgs.npz cond_lh.npz cond_hawkes.npz $(hyb cond) $(cop cond) \
       --refs cop_cond copg_cond ;;
  2) "$PY" -m wildfire.paper.joint_counts --config "$CONFIG" --part summary --tag _copula \
       --inputs counts_cgs.npz counts_lh.npz counts_hawkes.npz $(hyb counts) $(cop counts) \
       --refs cop_sim copg_sim --scen-inputs "${scen[@]}" --scen-refs cop_sim copg_sim ;;
  3) "$PY" -m wildfire.paper.joint_eval --config "$CONFIG" --part summary --tag _scenarios_copula \
       --refs cop_sim copg_sim --inputs "${scen[@]}" ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
