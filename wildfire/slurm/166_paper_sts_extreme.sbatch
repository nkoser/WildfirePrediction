#!/bin/bash
#SBATCH --job-name=wf_sts_extreme
#SBATCH --array=0-29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Extremlagen als Eingabe und Monotonie (wildfire/paper/sts_extreme.py), Protokoll wie die Ablation (74):
# Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke, EMA 0.999, Batch 32, lr 3e-4, 10 x 3000 Schritte, letzte
# Gewichte. Je Variante 3 Seeds (seed = i % 3, Variante v = i / 3), Tag _<name>_s<seed>, ein Seed je Job.
#   v0 lh_ref        LH ohne Zusatz (Nachbau der Referenz abl_lh mit diesem Code)
#   v1 lh_reg        + regionale Feueraktivitaet der Saison          (Punkt 2)
#   v2 lh_px         + Feuerwetter relativ zur Zelle, Kappung, NoData (Punkt 3)
#   v3 lh_both       + beides
#   v4 lh_mono       + Monotonie-Strafterm, Gewicht 1                (Punkt 5)
#   v5 lh_all        + beides + Monotonie
#   v6 cg_both       Contagion ohne Warmstart + beides
#   v7 cg_mono       Contagion ohne Warmstart + Monotonie
#   v8 cg_all        Contagion ohne Warmstart + beides + Monotonie
#   v9 cg_ref        Contagion ohne Warmstart ohne Zusatz (Nachbau von abl_cg_scratch)
# Braucht derived_data/ctx_features (165). Ausgaben NEU: checkpoints/paper/sts_extreme_<name>_s<seed>_s<seed>.pt,
# derived_data/calib_cache_ext/, Manifest p35_sts_extreme_<name>_s<seed>.
#   SMOKE=1: 200 Schritte x 1 Epoche, 3000 Trainings-/Auswertestichtage, Tag _smoke_<name>, CTX_DIR=ctx_features_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
case $v in
  0) n=lh_ref;  args=(--head lh --ctx none) ;;
  1) n=lh_reg;  args=(--head lh --ctx reg) ;;
  2) n=lh_px;   args=(--head lh --ctx px) ;;
  3) n=lh_both; args=(--head lh --ctx both) ;;
  4) n=lh_mono; args=(--head lh --ctx none --mono 1) ;;
  5) n=lh_all;  args=(--head lh --ctx both --mono 1) ;;
  6) n=cg_both; args=("${cg[@]}" --ctx both) ;;
  7) n=cg_mono; args=("${cg[@]}" --ctx none --mono 1) ;;
  8) n=cg_all;  args=("${cg[@]}" --ctx both --mono 1) ;;
  9) n=cg_ref;  args=("${cg[@]}" --ctx none) ;;
esac
small=()
if [ -n "${SMOKE:-}" ]; then
  small=(--epochs 1 --steps-per-epoch 200 --limit-train 3000 --limit-eval 3000); n=smoke_$n
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_extreme --config "$CONFIG" --ctx-dir "${CTX_DIR:-ctx_features}" \
    --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" "${small[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
