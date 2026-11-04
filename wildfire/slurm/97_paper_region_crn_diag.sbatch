#!/bin/bash
#SBATCH --job-name=wf_region_crn_diag
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Regionskurven, Diagnose (paper/region_eval_crn.py): Seed 0 auf 2000 Test-Stichtagen mit 16, 64 und 256
# Pfaden (gemeinsame Zufallszahlen) und 64 Pfaden ohne CRN wie region_eval. Frage: Ist der Abstand MC gegen
# Mean-Field beim einzelnen Pixel Rauschen oder Folge des Trainings auf Mean-Field-Randkurven?
#   0  Contagion warm aus STS-LH (_abl_cg, 74 v21)      -> region_crn_diag_cg.json
#   1  Contagion von Grund auf (_abl_cg_scratch, 74 v26) -> region_crn_diag_cg_scratch.json
# Braucht 85_paper_region_events. Checkpoints nur gelesen. Ausgabe NEU: metrics/paper_region/.
# Probelauf:  sbatch --array=0 --export=ALL,SMOKE=1 97_paper_region_crn_diag.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
case $i in
  0) args=(--cg "$CK/sts_cache_G_next_cube_abl_cg_s0_s0.pt" --tag _diag_cg) ;;
  1) args=(--cg "$CK/sts_cache_G_next_cube_abl_cg_scratch_s0_s0.pt" --tag _diag_cg_scratch) ;;
esac
args+=(--lh "$CK/kein_lh_in_diesem_lauf.pt" --n 2000 --Ms 16,64,256 --M-ind 64)
if [ -n "${SMOKE:-}" ]; then args+=(--n 200 --Ms 4,16 --M-ind 4 --boot 100 --tag _smoke$i); fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.region_eval_crn --config "$CONFIG" "${args[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
