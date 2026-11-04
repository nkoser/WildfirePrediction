#!/bin/bash
#SBATCH --job-name=wf_split_fev
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# focus_eval (wie 145) auf den Kandidaten-Splits: PR-AUC je dt auf allen Test-Stichproben und ign30 des Split-Caches,
# Modelle aus 234, logistische Feuerhistorie auf paper_pixel_<sp>. U-Net kommt in 237 aus den Vorhersagen von 226/230.
#   0 ys (cache_G_yearstrat)   1 bs (cache_G_blockstrat)
# Ausgaben NEU: metrics/focus_eval_<sp>${SFX}/, Manifest p45_focus_eval_<sp>${SFX}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
sps=(ys bs); caches=(cache_G_yearstrat cache_G_blockstrat); p=${sps[$i]}${SFX:-}
M="${p}_lh_ref ${p}_cg_ref ${p}_cg_mono ${p}_lh_fdrop:focus=fdrop ${p}_lh_resid:focus=resid ${p}_lh_ignw:focus=ignw"
M="$M ${p}_lh_fdropall:focus=fdropall ${p}_lh_fdrop_mono:focus=fdrop ${p}_lh_fdrop_mono_rf:focus=fdrop:pad=reflect"
M="$M ${p}_cg_fdrop:focus=fdrop ${p}_cg_ignw:focus=ignw ${p}_cg_fdrop_mono:focus=fdrop ${p}_cg_fdrop_mono_rf:focus=fdrop:pad=reflect"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.focus_eval --config "$CONFIG" --cache "${caches[$i]}" --cube-dir "paper_pixel_${sps[$i]}${SP_SFX:-}" \
    --tag "_${p}" --models $M ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
