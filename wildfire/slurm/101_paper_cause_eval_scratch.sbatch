#!/bin/bash
#SBATCH --job-name=wf_cause_eval_scr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Wie 89_paper_cause_eval, aber fuer die Ursachen-Modelle von Grund auf (91, _lam*_scratch) gegen
# _abl_cg_scratch und _abl_lh, auf denselben Stichtagen (gleiches --n und --seed wie 89).
# Braucht 85, 87 (eval) und 91. Alle Checkpoints nur gelesen.
# Ausgabe NEU: metrics/paper_cause/cause_eval_scratch.json (bricht ab, wenn sie existiert).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.contagion_cause_eval --config "$CONFIG" --tag _scratch --models \
  "cg:/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause_lam*_scratch_s?.pt" \
  "cg:/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt" \
  "lh:/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_lh_s?_s?.pt"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
