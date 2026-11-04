#!/bin/bash
#SBATCH --job-name=wf_imp_focus
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Permutation Importance (wildfire/paper/importance_v2.py, unveraendert; Kopie des Musters von 204 der anderen
# Sitzung) fuer lh_ref, cg_ref und die besten sts_focus-Varianten, Teilmengen all / ign30 / ign183.
# Ausgabe NEU: metrics/importance_v2/<name>_s<seed>${TAG}.json (TAG Standard _focus, kollidiert nicht mit 204).
#   0-2 lh_ref  3-5 cg_ref  6-8 ${N1} (Huelle ${W1:-none})  9-11 ${N2} (Huelle ${W2:-none})
#   AGGREGATE=1 TAG=_focus sbatch --array= --gres= 147_...sbatch    CSV + Abbildungen
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
C=/data/Wildfire/checkpoints/paper
TAG=${TAG:-_focus}
if [ -n "${AGGREGATE:-}" ]; then
    "$PY" -m wildfire.paper.importance_v2 --config "$CONFIG" --aggregate --tag "$TAG"
else
    i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
    v=$((i / 3)); s=$((i % 3))
    names=(lh_ref cg_ref "${N1:?}" "${N2:?}"); wraps=(none none "${W1:-none}" "${W2:-none}")
    n=${names[$v]}; w=${wraps[$v]}
    a=(--config "$CONFIG" --models "$n=$C/sts_extreme_${n}_s${s}_s${s}.pt" --index 0 --tag "$TAG" ${EXTRA:-})
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ "$w" = none ]; then
      "$PY" -m wildfire.paper.importance_v2 "${a[@]}"
    else
      "$PY" -m wildfire.paper.focus_patch --target importance --wrap "$w" -- "${a[@]}"
    fi
fi
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
