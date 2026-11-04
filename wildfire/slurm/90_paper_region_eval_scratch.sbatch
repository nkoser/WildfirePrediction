#!/bin/bash
#SBATCH --job-name=wf_region_eval_scr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Wie 86_paper_region_eval, aber fuer Contagion-STS von Grund auf (_abl_cg_scratch, 74 v26) statt
# warm aus STS-LH. STS-LH wird hier nicht noch einmal gerechnet (steht in 86).
# Braucht 85_paper_region_events. Ausgabe NEU: metrics/paper_region/region_eval_cg_scratch<...>.json
# Probelauf:  SMOKE=1 sbatch 90_paper_region_eval_scratch.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
args=(--cg "$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt" --lh "$CK/kein_lh_in_diesem_lauf.pt" --tag _cg_scratch)
if [ -n "${SMOKE:-}" ]; then
  args+=(--n 2000 --M 16 --tag _cg_scratch_smoke)
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.region_eval --config "$CONFIG" "${args[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
