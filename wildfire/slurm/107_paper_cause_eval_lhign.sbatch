#!/bin/bash
#SBATCH --job-name=wf_cause_eval_lhign
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Wie 89_paper_cause_eval, fuer STS-LH trainiert auf die erste NFDB-Zuendung (104 v6/v7, lh_ign warm und von
# Grund auf), auf denselben Stichtagen wie 89 (gleiches --n und --seed). Bias wie dort (load_sts), damit die
# Zahlen neben cause_eval.json stehen koennen. Braucht 104. Ausgabe NEU: metrics/paper_cause/cause_eval_lhign.json.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CJ=/data/Wildfire/checkpoints/paper_joint
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_cause_eval --config "$CONFIG" --tag _lhign --models \
  "lh:$CJ/sts_cache_G_next_cube_lhign_warm_s?.pt" "lh:$CJ/sts_cache_G_next_cube_lhign_scratch_s?.pt"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
