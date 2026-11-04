#!/bin/bash
#SBATCH --job-name=wf_sts_spread_tfmf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS als Simulator: Teacher Forcing + kleines Mean-Field-Gewicht (Mitte zwischen cg_ell11tf10 = MF 1
# und cg_tfonly = MF 0), Ellipse 11 x 11, gleichverteilte Stichtage, von Grund auf (wildfire/paper/sts_spread_tfonly.py).
# Je Variante 3 Seeds (seed = i % 3, v = i / 3), Tag _<name>_s<seed>:
#   v0 cg_tfmf01   TF lam 1.0, Mean-Field-Gewicht 0.1
#   v1 cg_tfmf03   TF lam 1.0, Mean-Field-Gewicht 0.3
# Ausgaben NEU: checkpoints/paper/sts_spread_tfonly_<name>_s<seed>_s<seed>.pt, Manifest p63_sts_spread_tfonly_<name>_s<seed>.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3))
case $v in
  0) n=cg_tfmf01; args=(--spread ell --cg-kernel 11 --tf-weight 1.0 --mf-weight 0.1 --big-share 0) ;;
  1) n=cg_tfmf03; args=(--spread ell --cg-kernel 11 --tf-weight 1.0 --mf-weight 0.3 --big-share 0) ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_spread_tfonly --config "$CONFIG" --seeds $seed --tag "_${n}_s${seed}" "${args[@]}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
