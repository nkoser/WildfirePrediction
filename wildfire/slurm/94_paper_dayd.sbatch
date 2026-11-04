#!/bin/bash
#SBATCH --job-name=wf_dayd
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll mit dem Cache-Label (Feuer im Ausschnitt an D, --target cache)
# statt D+1, mit und ohne MODIS-Feuer als Eingabe. Gegenstueck zu 85 (D+1).
# Die U-Nets auf Ziel D gibt es schon: _paperhp, _paperhp_no_fire, _paperhp_no_modis.
# Index i: Variante i / 3, Seed i % 3. Alles neue Namen (Ziel D: kein _next).
#   v0  LH d 64, EMA, Auswahl NLL wie 68 v5, paper55   -> p15_sts_dctm_cache_G_cube_d64_lh_ema_nll_s<seed>
#   v1  dasselbe, no_fire                             -> p15_sts_dctm_cache_G_no_fire_cube_d64_lh_ema_nll_s<seed>
#   v2  dasselbe, no_modis                            -> p15_sts_dctm_cache_G_no_modis_cube_d64_lh_ema_nll_s<seed>
#   v3  Contagion v2 von Grund auf wie 83 v1, paper55 -> p15_sts_dctm_cache_G_cube_d64_cg2_ema_nll_s<seed>
# Contagion nur mit paper55: der Ausbreitungsterm liest das beobachtete Feuer direkt.
# Tabelle: 96_paper_dayd_table.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
base=(--cache cache_G --train-source cube --target cache --steps-per-epoch 3000 --ema 0.999
      --d 64 --depth 4 --select survnll)
case $v in
  0) args=("${base[@]}" --head lh --epochs 10 --channels paper55 --tag "_d64_lh_ema_nll_s${seed}") ;;
  1) args=("${base[@]}" --head lh --epochs 10 --channels no_fire --tag "_d64_lh_ema_nll_s${seed}") ;;
  2) args=("${base[@]}" --head lh --epochs 10 --channels no_modis --tag "_d64_lh_ema_nll_s${seed}") ;;
  3) args=("${base[@]}" --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --epochs 15
           --tag "_d64_cg2_ema_nll_s${seed}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --seeds $seed --save-ckpt "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
