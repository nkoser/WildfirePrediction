#!/bin/bash
#SBATCH --job-name=wf_region_crn
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Regionskurven, volle Auswertung mit gemeinsamen Zufallszahlen und gepaartem Bootstrap
# (paper/region_eval_crn.py): 20000 Test-Stichtage, alle 3 Seeds, STS-LH auf denselben Stichtagen.
#   0  Contagion warm aus STS-LH (_abl_cg)          -> region_crn_cg.json (+ _nll.npz)
#   1  Contagion von Grund auf (_abl_cg_scratch)    -> region_crn_cg_scratch.json (+ _nll.npz)
# Pfadzahl ueber MS (Standard 64), nach der Diagnose 97 festlegen. Braucht 85. Ausgabe NEU: metrics/paper_region/.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
case $i in
  0) args=(--cg "$CK/sts_cache_G_next_cube_abl_cg_s?_s?.pt" --tag _cg) ;;
  1) args=(--cg "$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt" --tag _cg_scratch) ;;
esac
args+=(--lh "$CK/sts_cache_G_next_cube_abl_lh_s?_s?.pt" --n 20000 --Ms "${MS:-64}")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.region_eval_crn --config "$CONFIG" "${args[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
