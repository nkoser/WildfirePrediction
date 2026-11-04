#!/bin/bash
#SBATCH --job-name=wf_calib_resume_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Rauchtest fuer joint_calib --resume-log / --skip-existing: wiederholt den Contagion-Fit aus 108 (Job 4831) mit
# denselben Einstellungen und uebernimmt alle Punkte aus dessen Log; Ergebnis muss gleich calib_smoke2 sein.
# Ausgabe NEU: metrics/paper_joint/calib_smoke3/.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
L=/scratch/shared/nik/Wildfire/logs
set -e
a=(--config "$CONFIG" --part fit --calib-dir calib_smoke3 --n-val 256 --J 4 --M 2 --rows 4096 --cg-line -2 0
   --cg-step 0.5 --cg-min-step 0.5 --models "cg=cg:$CK/${G}_abl_cg_s0_s0.pt")
"$PY" -m wildfire.paper.joint_calib "${a[@]}" --resume-log "$L/wf_calib_smoke2_4831.out"
"$PY" -m wildfire.paper.joint_calib "${a[@]}" --skip-existing          # zweiter Aufruf: nur ueberspringen
echo "[$(date -u +%FT%TZ)] exit 0"
