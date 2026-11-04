#!/bin/bash
#SBATCH --job-name=wf_joint_train
#SBATCH --array=0-23%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Training fuer die Experimente zur gemeinsamen Verteilung (paper/joint_train.py). Index i: Variante i / 3,
# Seed i % 3. warm: Rumpf aus _abl_lh, 3 x 3.000 Schritte; scratch: ohne Warmstart, 10 x 3.000 Schritte.
#   v0  cg_tf warm             Contagion, exakte gemeinsame Likelihood   -> ..._cgtf_warm_s<seed>
#   v1  cg_tf scratch                                                     -> ..._cgtf_scratch_s<seed>
#   v2  lh_region mask warm    STS-LH + Maskenkopf                        -> ..._lhreg_mask_warm_s<seed>
#   v3  lh_region mask scratch                                            -> ..._lhreg_mask_scratch_s<seed>
#   v4  lh_region fixed warm   STS-LH + Koepfe sq5, sq9, nw6              -> ..._lhreg_fixed_warm_s<seed>
#   v5  lh_region fixed scratch                                           -> ..._lhreg_fixed_scratch_s<seed>
#   v6  lh_ign warm            STS-LH auf erste NFDB-Zuendung             -> ..._lhign_warm_s<seed>
#   v7  lh_ign scratch                                                    -> ..._lhign_scratch_s<seed>
# Braucht 74 (v0, _abl_lh) und 87 (train_cause.npz, fuer lh_ign). Ausgabe NEU: checkpoints/paper_joint/.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
warm=(--epochs 3 --tag _warm)
scratch=(--init-from= --epochs 10 --tag _scratch)
case $v in
  0) args=(--model cg_tf "${warm[@]}") ;;
  1) args=(--model cg_tf "${scratch[@]}") ;;
  2) args=(--model lh_region --region-mode mask "${warm[@]}") ;;
  3) args=(--model lh_region --region-mode mask "${scratch[@]}") ;;
  4) args=(--model lh_region --region-mode fixed "${warm[@]}") ;;
  5) args=(--model lh_region --region-mode fixed "${scratch[@]}") ;;
  6) args=(--model lh_ign "${warm[@]}") ;;
  7) args=(--model lh_ign "${scratch[@]}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.joint_train --config "$CONFIG" --seed $seed "${args[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
