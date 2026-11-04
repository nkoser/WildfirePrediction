#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_split_prep
#SBATCH --array=0-21%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Stufe 1 der Portierung aller Paper-Analysen auf die Kandidaten-Splits: fehlende Vorstufen-Daten je Split, nur CPU.
# Alle Builder unveraendert, Manifest-Namen ueber split_run mit Suffix _<sp> (sonst ersetzt z. B. rate_labels das
# zeitliche p23_rate_labels). Split sp = i / 11 (0 ys = year_strat, 1 bs = block_strat), Aufgabe k = i % 11:
#   0 km_meta.npz in surv_eval_<sp> nur aus dem train-Teil (split_km)       -> surv_eval, horizon, state_eval
#   1 train_rate.npz in paper_pixel_<sp> (rate_labels, wie 73)               -> Ablation v6/v7/v27
#   2 train_Xf/kf in paper_pixel_<sp> (future_weather, wie 78)               -> Oracle v31, Teaser-Oracle
#   3 paper_pixel_<sp>_ring5 (ring_targets, wie 250)                         -> sts_spread_tfonly
#   4 paper_pixel_<sp>_future31 (future_maps, wie 252)                       -> fire_gpt
#   5 region_eval_<sp>/pixel_events.npz (split_pixel_events)                -> cause eval, multistate_eval
#   6 cause_labels/train_cause_<sp> (cause_labels --part train, wie 87 v0)   -> Multi-State v1
#   7-10 state_features/<teil>_<sp> fuer train/eval/val/paper (wie 200)       -> Multi-State v2
# Danach 277 (cause eval + state_labels, braucht 5). Rauchtest: SFX=_smoke beschraenkt k 7-10 auf 2.000 Stichtage.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
k=$((i % 11)); s=$((i / 11))
sps=(ys bs); names=(year_strat block_strat); caches=(cache_G_yearstrat cache_G_blockstrat)
sp=${sps[$s]}; cube=paper_pixel_${sp}; surv=surv_eval_${sp}
run=("$PY" -m wildfire.paper.split_run --manifest-suffix "_${sp}${SFX:-}")
case $k in
  0)  cmd=("$PY" -m wildfire.paper.split_km --config "$CONFIG" --name "${names[$s]}" --out-surv "$surv") ;;
  1)  cmd=("${run[@]}" wildfire.paper.rate_labels --config "$CONFIG" --cube-dir "$cube") ;;
  2)  cmd=("${run[@]}" wildfire.paper.future_weather --config "$CONFIG" --cube-dir "$cube") ;;
  3)  cmd=("${run[@]}" wildfire.paper.ring_targets --config "$CONFIG" --cube-dir "$cube") ;;
  4)  cmd=("${run[@]}" wildfire.paper.future_maps --config "$CONFIG" --cube-dir "$cube") ;;
  5)  cmd=("$PY" -m wildfire.paper.split_pixel_events --config "$CONFIG" --surv-dir "$surv" --out-dir "region_eval_${sp}") ;;
  6)  cmd=("${run[@]}" wildfire.paper.cause_labels --config "$CONFIG" --part train --cube-dir "$cube" --tag "_${sp}") ;;
  *)  parts=(train eval val paper); p=${parts[$((k - 7))]}
      a=(--keys raw); [ -n "${SFX:-}" ] && a+=(--max-n 2000)
      cmd=("${run[@]}" wildfire.paper.state_features --config "$CONFIG" --part "$p" --cube-dir "$cube"
           --surv-dir "$surv" --cache "${caches[$s]}" --tag "_${sp}${SFX:-}" "${a[@]}") ;;
esac
echo "[split_prep] $sp Aufgabe $k: ${cmd[*]}"
"${cmd[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
