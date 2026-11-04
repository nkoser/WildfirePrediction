#!/bin/bash
#SBATCH --job-name=wf_stress
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Stresstest "Was waere wenn" (wildfire/paper/stress_test.py): fertige Modelle des Ablationsprotokolls
# auf den Test-Stichtagen von surv_eval_test mit kuenstlich verschaerftem Feuerwetter (Verschiebung,
# feste Werte bis 320, FWI/ISI-Code 255). Kein Neutraining. Je Array-Index ein Checkpoint:
#   0-2  lh          sts_cache_G_next_cube_abl_lh_s{0,1,2}
#   3-5  cg_scratch  sts_cache_G_next_cube_abl_cg_scratch_s{0,1,2}
# Ausgabe NEU: derived_data/stress_test/<modell>_s<seed>${TAG}.npz; danach 161 (Zusammenfassung).
#   Rauchtest: TAG=_smoke EXTRA="--limit 256" sbatch --array=0 160_paper_stress_test.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
names=(lh lh lh cg_scratch cg_scratch cg_scratch)
seed=$((i % 3)); name=${names[$i]}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.stress_test --config "$CONFIG" \
    --ckpt "$CK/sts_cache_G_next_cube_abl_${name}_s${seed}_s${seed}.pt" --model-name "$name" \
    --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
