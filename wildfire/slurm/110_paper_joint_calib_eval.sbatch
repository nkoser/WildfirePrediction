#!/bin/bash
#SBATCH --job-name=wf_calib_eval
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Formen und Szenarien wie 105, aber mit den nachkalibrierten Modellen (joint_calib --part shapes|scenarios,
# Kalibrierung aus metrics/paper_joint/calib/, 109). 20.000 Test-Stichtage, 32 Pfade wie 105.
#   0  Formen cg        1  Formen cgs        2  Formen lh + lh_region (4 Familien)
#   3  Szenarien cg     4  Szenarien cgs     5  Szenarien lh
# Ausgabe NEU: metrics/paper_joint/{shapes,scenarios}_<name>_calib.{json,npz}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
CJ=/data/Wildfire/checkpoints/paper_joint
G=sts_cache_G_next_cube
cg="cg=cg:$CK/${G}_abl_cg_s?_s?.pt"
cgs="cgs=cg:$CK/${G}_abl_cg_scratch_s?_s?.pt"
lh="lh=lh:$CK/${G}_abl_lh_s?_s?.pt"
lhr=("lhrm_w=lh:$CJ/${G}_lhreg_mask_warm_s?.pt" "lhrm_s=lh:$CJ/${G}_lhreg_mask_scratch_s?.pt"
     "lhrf_w=lh:$CJ/${G}_lhreg_fixed_warm_s?.pt" "lhrf_s=lh:$CJ/${G}_lhreg_fixed_scratch_s?.pt")
case $i in
  0) a=(--part shapes --tag _cg_calib --models "$cg") ;;
  1) a=(--part shapes --tag _cgs_calib --models "$cgs") ;;
  2) a=(--part shapes --tag _lh_calib --models "$lh" "${lhr[@]}") ;;
  3) a=(--part scenarios --tag _cg_calib --models "$cg") ;;
  4) a=(--part scenarios --tag _cgs_calib --models "$cgs") ;;
  5) a=(--part scenarios --tag _lh_calib --models "$lh") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.joint_calib --config "$CONFIG" --calib-dir calib --M 32 "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
