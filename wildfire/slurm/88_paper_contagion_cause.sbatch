#!/bin/bash
#SBATCH --job-name=wf_cg_cause
#SBATCH --array=0-11%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS mit teilweise beobachteter Ursache (paper/contagion_cause.py), Einstellungen wie
# 74_paper_ablation v21 (_abl_cg). Braucht 87_paper_cause_labels (train). Index i: Variante i / 3, Seed i % 3.
#   v0  lambda 0      Kontrolle, muss _abl_cg reproduzieren      -> ..._cause_lam0_s<seed>
#   v1  lambda 1      Zuendung + Ausbreitung                     -> ..._cause_lam1_s<seed>
#   v2  lambda 0.3    Zuendung + Ausbreitung                     -> ..._cause_lam03_s<seed>
#   v3  lambda 1      nur Zuendungs-Labels                       -> ..._cause_lam1_ign_s<seed>
# Ausgabe NEU: checkpoints/paper_cause/ (bricht ab, wenn der Checkpoint existiert).
# Rauchtest:  sbatch --array=0 --export=ALL,SMOKE=1 88_paper_contagion_cause.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) args=(--lam 0 --tag _lam0) ;;
  1) args=(--lam 1 --tag _lam1) ;;
  2) args=(--lam 0.3 --tag _lam03) ;;
  3) args=(--lam 1 --use-spread 0 --tag _lam1_ign) ;;
esac
if [ -n "${SMOKE:-}" ]; then
  args=(--lam 1 --limit 2000 --epochs 1 --steps-per-epoch 20 --tag _smoke)
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_cause --config "$CONFIG" --seed $seed "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
