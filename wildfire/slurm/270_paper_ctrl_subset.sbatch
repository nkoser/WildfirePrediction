#!/bin/bash
#SBATCH --job-name=wf_ctrl_sub
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Status-Versuch nur mit Wuerfel-Stichtagen ab 2011 (wildfire/paper/ctrl_subset.py), aus den Kontexten von 254.
# Aufgabe i: 0 t, 1 ys, 2 bs. Ausgaben NEU: derived_data/ctx_ctrl11{,_ys,_bs} (Status) und ctx_zero11{,_ys,_bs}
# (Nullen, Referenz auf derselben Teilmenge), Manifest p55_ctrl_subset_*. Danach 271 (Training).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
sfx=("" "_ys" "_bs"); cubes=(paper_pixel paper_pixel_ys paper_pixel_bs)
"$PY" -m wildfire.paper.ctrl_subset --config "$CONFIG" --src "ctx_ctrl${sfx[$i]}" --cube-dir "${cubes[$i]}" \
    --out "ctx_ctrl11${sfx[$i]}${SFX:-}" --zero-out "ctx_zero11${sfx[$i]}${SFX:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
