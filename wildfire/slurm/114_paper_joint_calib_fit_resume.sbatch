#!/bin/bash
#SBATCH --job-name=wf_calib_fit_resume
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Fortsetzung von 109 (Job 4832, 12-h-Limit zu knapp: ~31 min je Zielwert): gleiche Modelle und Einstellungen,
# uebernimmt alle schon berechneten Punkte aus den 4832-Logs (gleiche Zufallszahlen) und ueberspringt Modelle, deren
# Kalibrierung 4832 schon geschrieben hat. Nach 4832 starten (afterany). Ausgabe: metrics/paper_joint/calib/<ckpt>.json.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
CJ=/data/Wildfire/checkpoints/paper_joint
G=sts_cache_G_next_cube
case $i in
  0|1|2) m=("cg=cg:$CK/${G}_abl_cg_s${i}_s${i}.pt") ;;
  3|4|5) s=$((i - 3)); m=("cgs=cg:$CK/${G}_abl_cg_scratch_s${s}_s${s}.pt") ;;
  6)     m=("lh=lh:$CK/${G}_abl_lh_s?_s?.pt" "lhrm_w=lh:$CJ/${G}_lhreg_mask_warm_s?.pt"
            "lhrm_s=lh:$CJ/${G}_lhreg_mask_scratch_s?.pt" "lhrf_w=lh:$CJ/${G}_lhreg_fixed_warm_s?.pt"
            "lhrf_s=lh:$CJ/${G}_lhreg_fixed_scratch_s?.pt") ;;
esac
logs=(/scratch/shared/nik/Wildfire/logs/wf_calib_fit_4832_*.out)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.joint_calib --config "$CONFIG" --part fit --calib-dir calib --models "${m[@]}" \
  --skip-existing --resume-log "${logs[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
