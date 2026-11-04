#!/bin/bash
#SBATCH --job-name=wf_cg_cause_scr
#SBATCH --array=0-11%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Wie 88_paper_contagion_cause, aber von Grund auf ohne Warmstart aus STS-LH, mit 10 Epochen wie
# 74_paper_ablation v26 (_abl_cg_scratch). Index i: Variante i / 3, Seed i % 3.
#   v0  lambda 0      Kontrolle, muss _abl_cg_scratch reproduzieren  -> ..._cause_lam0_scratch_s<seed>
#   v1  lambda 1      Zuendung + Ausbreitung                         -> ..._cause_lam1_scratch_s<seed>
#   v2  lambda 0.3    Zuendung + Ausbreitung                         -> ..._cause_lam03_scratch_s<seed>
#   v3  lambda 1      nur Zuendungs-Labels                           -> ..._cause_lam1_ign_scratch_s<seed>
# Braucht 87_paper_cause_labels (train). Ausgabe NEU: checkpoints/paper_cause/ (bricht ab, wenn vorhanden).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
scratch=(--init-from= --epochs 10)
case $v in
  0) args=(--lam 0 --tag _lam0_scratch) ;;
  1) args=(--lam 1 --tag _lam1_scratch) ;;
  2) args=(--lam 0.3 --tag _lam03_scratch) ;;
  3) args=(--lam 1 --use-spread 0 --tag _lam1_ign_scratch) ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_cause --config "$CONFIG" --seed $seed "${scratch[@]}" "${args[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
