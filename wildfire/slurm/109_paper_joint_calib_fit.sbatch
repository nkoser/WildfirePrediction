#!/bin/bash
#SBATCH --job-name=wf_calib_fit
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Nachkalibrierung (paper/joint_calib.py --part fit) auf 10.000 Val-Stichtagen, 16 Pixel je Stichtag, 16 Pfade.
#   0-2  Contagion warm (_abl_cg) Seed 0-2           3-5  Contagion von Grund auf (_abl_cg_scratch) Seed 0-2
#   6    STS-LH (_abl_lh) und die lh_region-Modelle (Pixel- und Ausschnittskopf; Regionskoepfe unveraendert)
# Braucht derived_data/joint_calib/val_pixel_events.npz (108). Ausgabe NEU: metrics/paper_joint/calib/<ckpt>.json.
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.joint_calib --config "$CONFIG" --part fit --calib-dir calib --models "${m[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
