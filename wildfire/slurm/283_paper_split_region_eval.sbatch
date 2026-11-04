#!/bin/bash
#SBATCH --job-name=wf_split_region
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Regionsauswertung (Deck: "Same Marginals, Better Joint Forecasts") auf den Kandidaten-Splits, wie 86/90 auf t:
# STS-LH (74 v0) und Contagion-STS von Grund auf (74 v26 cg_scratch) des jeweiligen Splits, 20.000 Test-Stichtage, M 64.
# region_eval liest fest derived_data/region_eval/pixel_events.npz (zeitlicher Split) -> per split_run auf
# region_eval_<sp>/pixel_events.npz (276 k5) umgelenkt. i = 0 ys, 1 bs.
# Braucht 74 (5812/5813) Index 0-2 und 78-80. Ausgabe NEU: metrics/paper_region/region_eval_<sp>.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
sps=(ys bs); caches=(cache_G_yearstrat cache_G_blockstrat); sp=${sps[$i]}; cache=${caches[$i]}
CK=/data/Wildfire/checkpoints/paper/sts_${cache}_next_cube
out=/data/Wildfire/metrics/paper_region/region_eval_${sp}.json
if [ -e "$out" ]; then echo "ABBRUCH: $out existiert schon"; exit 1; fi
for p in "${CK}_abl_lh_s?_s?.pt" "${CK}_abl_cg_scratch_s?_s?.pt"; do
  [ "$(ls $p 2>/dev/null | wc -l)" -eq 3 ] || { echo "ABBRUCH: nicht 3 Checkpoints fuer $p"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.split_run --manifest-suffix "_${sp}" \
    --redirect "region_eval/pixel_events.npz=region_eval_${sp}/pixel_events.npz" \
    wildfire.paper.region_eval --config "$CONFIG" --surv-dir "surv_eval_${sp}" \
    --cg "${CK}_abl_cg_scratch_s?_s?.pt" --lh "${CK}_abl_lh_s?_s?.pt" --tag "_${sp}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
