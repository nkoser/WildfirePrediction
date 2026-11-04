#!/bin/bash
#SBATCH --job-name=wf_split_sxt
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# sts_extreme_table (wie 168) fuer die Laeufe aus 234: Survival-Metriken (NLL, O/E@30 je Testjahr), kalibriert auf
# Val des Splits. 0 ys (surv_eval_ys)  1 bs (surv_eval_bs). Ausgaben NEU: metrics/sts_extreme_<sp>${SFX}/,
# calib_cache_ext/ens_<sp>_<name>.npz, Manifest p36_sts_extreme_table_<sp>${SFX}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
sps=(ys bs); p=${sps[$i]}${SFX:-}
N=""
for n in lh_ref lh_fdrop lh_resid lh_ignw lh_fdropall lh_fdrop_mono lh_fdrop_mono_rf cg_ref cg_mono cg_fdrop cg_ignw cg_fdrop_mono cg_fdrop_mono_rf; do
  N="$N ${p}_$n"
done
"$PY" -m wildfire.paper.sts_extreme_table --config "$CONFIG" --names $N --refs --surv-dir "surv_eval_${sps[$i]}${SP_SFX:-}" \
    --tag "_${p}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
