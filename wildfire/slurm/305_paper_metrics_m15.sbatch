#!/bin/bash
#SBATCH --job-name=wf_metrics_m15
#SBATCH --array=0,4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# paper_metrics_ckpt wie in 75 (nur dieser Teil, ohne surv_eval), aber ECE/MCE mit 15 Bins
# (wildfire/paper/m15_run.py). Manifest p17_paper_metrics_cache_G_next_abl_g<n>_m15; die alten
# Manifeste mit 10 Bins bleiben. Der Nachbau (rec, --rec-run Standard) wird mitbewertet.
#   Array wie 75: 0 = lh/bce-Koepfe, 4 = Contagion (cg ...); andere Gruppen per --array=...
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
g=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper/sts_${CACHE:-cache_G}_next
groups=("lh bce bce2 bce8 bce16 bce31" "lh_rate1 lh_rate5 hist7 hist14 random fire"
        "paperonly dctm cox deephit joint stem3d" "frac10 frac25 frac50 frac100"
        "cg cg_k1 cg_k3 cg_obs7" "cg_geo cg_scratch cg_rate1" "paperonly_last paperonly_prauc")
models=()
for n in ${groups[$g]}; do
  src=cube; case $n in paperonly*) src=cache ;; esac
  models+=("sts_abl_${n}:${CK}_${src}_abl_${n}_s?_s?.pt")
done
echo "[m15] Gruppe $g: ${models[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.m15_run --manifest-suffix _m15 wildfire.paper.paper_metrics_ckpt \
    --config "$CONFIG" --cache "${CACHE:-cache_G}" --models "${models[@]}" --tag "_abl_g${g}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
