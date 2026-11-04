#!/bin/bash
#SBATCH --job-name=wf_abl_eval
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung der Ablation (74): je Gruppe surv_eval (100.000 zufaellige Stichtage 2023-2025,
# getrennt nach Feuer im Fenster) und paper_metrics_ckpt (Paper-Protokoll, mit Rate-Kopf und
# rohen bce-Ausgaengen). Manifeste p16_surv_eval_abl_g<n>, p17_paper_metrics_cache_G_next_abl_g<n>.
# Flaechentreu: CACHE=cache_G_ea SURV_DIR=surv_eval_test_ea AW=1 TAGX=_ea sbatch ...
# Originalpixel gepaart dazu: AW=1 SUBSET=cache_G_ea TAGX=_orig sbatch ...
#   AW=1      surv_eval zusaetzlich je km^2 (--area-weight)
#   SUBSET    paper_metrics_ckpt nur auf den Stichproben mit Fenster im Wuerfel (ok.npy dieses Caches)
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
echo "[abl-eval] Gruppe $g: ${models[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name "${SURV_DIR:-surv_eval_test}" ${AW:+--area-weight} \
    --subsets --models "${models[@]}" --tag "_abl_g${g}${TAGX:-}" || exit 1
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache "${CACHE:-cache_G}" ${SUBSET:+--subset "$SUBSET"} \
    --models "${models[@]}" --tag "_abl_g${g}${TAGX:-}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
