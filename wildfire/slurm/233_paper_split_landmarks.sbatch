#!/bin/bash
#SBATCH --job-name=wf_split_lm
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Wuerfel-Stichtage je Kandidaten-Split (wildfire/paper/split_landmarks.py) fuer die Ablation sts_extreme/sts_focus:
#   0  year_strat   -> derived_data/paper_pixel_ys,  surv_eval_ys
#   1  block_strat  -> derived_data/paper_pixel_bs,  surv_eval_bs
# Training 240.000 Stichtage (120k fire, 120k random) wie paper_pixel, Val 50.000, Test 100.000 wie surv_eval_test.
# Rauchtest: SFX=_smoke EXTRA="--n-fire 2000 --n-random 2000 --n-val 1000 --n-eval 1000"
# Ausgaben NEU, Manifest p50_split_landmarks_<name>${SFX}. Danach 234 (Training).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
names=(year_strat block_strat); short=(ys bs)
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
"$PY" -m wildfire.paper.split_landmarks --config "$CONFIG" --name "${names[$i]}" \
    --out-cube "paper_pixel_${short[$i]}${SFX:-}" --out-surv "surv_eval_${short[$i]}${SFX:-}" --tag "${SFX:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
