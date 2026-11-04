#!/bin/bash
#SBATCH --job-name=wf_sts_spread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS mit gerichtetem, wetterabhaengigem Kern (wildfire/paper/sts_spread.py, contagion_spread.py).
# Protokoll wie sts_extreme cg_ref (166 v9): Wuerfel-Stichtage, Ziel D+1, ohne Warmstart, 10 x 3000 Schritte.
# Je Variante 3 Seeds (seed = i % 3, Variante v = i / 3), Tag _<name>_s<seed>:
#   v0 cg_k11free   nur groesserer Kern (11 x 11), frei            -> Reichweite
#   v1 cg_ell7      7 x 7 + Richtung (Wind des letzten Tages) und Reichweite
#   v2 cg_ell11     11 x 11 + Richtung und Reichweite
#   v3 cg_ell11fw   wie v2, Kern je Tag aus dem beobachteten Folgewetter (Oracle, obere Schranke; braucht 78)
# Referenz: sts_extreme_cg_ref_s* (166 v9). Ausgaben NEU: checkpoints/paper/sts_spread_<name>_s<seed>_s<seed>.pt,
# derived_data/calib_cache_ext/, Manifest p50_sts_spread_<name>_s<seed>.
#   SMOKE=1: 200 Schritte x 1 Epoche, 3000 Trainings-/Auswertestichtage, Tag _smoke_<name>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) n=cg_k11free;  args=(--spread free --cg-kernel 11) ;;
  1) n=cg_ell7;     args=(--spread ell --cg-kernel 7) ;;
  2) n=cg_ell11;    args=(--spread ell --cg-kernel 11) ;;
  3) n=cg_ell11fw;  args=(--spread ell --cg-kernel 11 --future-days 30) ;;
esac
small=()
if [ -n "${SMOKE:-}" ]; then
  small=(--epochs 1 --steps-per-epoch 200 --limit-train 3000 --limit-eval 3000); n=smoke_$n
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_spread --config "$CONFIG" --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" "${small[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
