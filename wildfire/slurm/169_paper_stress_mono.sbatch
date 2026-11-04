#!/bin/bash
#SBATCH --job-name=wf_stress_mono
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Stresstest wie 160 (wildfire/paper/stress_test.py), aber auf den Monotonie-Modellen aus sts_extreme
# (--mono, ohne Zusatzeingaben, Laeufe 166). Frage: verschwindet die FWI/ISI-Umkehr und der Abfall von
# F30 je Stichtag unter Verschiebung? Je Array-Index ein Checkpoint:
#   0-2  lh_mono  sts_extreme_lh_mono_s{0,1,2}_s{0,1,2}
#   3-5  cg_mono  sts_extreme_cg_mono_s{0,1,2}_s{0,1,2}
# Ausgabe NEU: derived_data/stress_test/<modell>_s<seed>${TAG}.npz (TAG Standard _mono); danach
#   MODELS="lh_mono cg_mono" TAG=_mono sbatch --dependency=afterok:<id> 161_paper_stress_table.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
names=(lh_mono lh_mono lh_mono cg_mono cg_mono cg_mono)
seed=$((i % 3)); name=${names[$i]}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.stress_test --config "$CONFIG" \
    --ckpt "$CK/sts_extreme_${name}_s${seed}_s${seed}.pt" --model-name "$name" \
    --tag "${TAG:-_mono}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
