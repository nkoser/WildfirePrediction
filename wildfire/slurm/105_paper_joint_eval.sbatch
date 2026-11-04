#!/bin/bash
#SBATCH --job-name=wf_joint_eval
#SBATCH --array=0-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung der gemeinsamen Verteilung (paper/joint_eval.py), 20.000 Test-Stichtage, 32 Pfade.
# Bias trained = wie im Training; default = wie load_sts (bisherige Auswertungen), zum Vergleich.
#   Formen (shapes)                                  Szenarien (scenarios)
#    0  cg       _abl_cg            trained           7  cg       _abl_cg            trained
#    1  cgs      _abl_cg_scratch    trained           8  cgs      _abl_cg_scratch    trained
#    2  cgtf_w   cg_tf warm         trained  (104)    9  cgtf_w   cg_tf warm         trained  (104)
#    3  cgtf_s   cg_tf scratch      trained  (104)   10  cgtf_s   cg_tf scratch      trained  (104)
#    4  lh + lh_region (4 Familien) trained  (104)   11  lh       _abl_lh            trained
#    5  cg       _abl_cg            default          12  cg       _abl_cg            default
#    6  lh       _abl_lh            default          13  lh       _abl_lh            default
# Ohne 104: --array=0,1,5,6,7,8,11,12,13; nach 104: --array=2,3,4,9,10.
# Ausgabe NEU: metrics/paper_joint/{shapes,scenarios}_<name>_<bias>.{json,npz}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
CJ=/data/Wildfire/checkpoints/paper_joint
G=sts_cache_G_next_cube
cg="cg=cg:$CK/${G}_abl_cg_s?_s?.pt"
cgs="cgs=cg:$CK/${G}_abl_cg_scratch_s?_s?.pt"
cgtf_w="cgtf_w=cg:$CJ/${G}_cgtf_warm_s?.pt"
cgtf_s="cgtf_s=cg:$CJ/${G}_cgtf_scratch_s?.pt"
lh="lh=lh:$CK/${G}_abl_lh_s?_s?.pt"
lhr=("lhrm_w=lh:$CJ/${G}_lhreg_mask_warm_s?.pt" "lhrm_s=lh:$CJ/${G}_lhreg_mask_scratch_s?.pt"
     "lhrf_w=lh:$CJ/${G}_lhreg_fixed_warm_s?.pt" "lhrf_s=lh:$CJ/${G}_lhreg_fixed_scratch_s?.pt")
case $i in
  0)  a=(--part shapes --bias trained --tag _cg_trained --models "$cg") ;;
  1)  a=(--part shapes --bias trained --tag _cgs_trained --models "$cgs") ;;
  2)  a=(--part shapes --bias trained --tag _cgtf_w_trained --models "$cgtf_w") ;;
  3)  a=(--part shapes --bias trained --tag _cgtf_s_trained --models "$cgtf_s") ;;
  4)  a=(--part shapes --bias trained --tag _lh_trained --models "$lh" "${lhr[@]}") ;;
  5)  a=(--part shapes --bias default --tag _cg_default --models "$cg") ;;
  6)  a=(--part shapes --bias default --tag _lh_default --models "$lh") ;;
  7)  a=(--part scenarios --bias trained --tag _cg_trained --models "$cg") ;;
  8)  a=(--part scenarios --bias trained --tag _cgs_trained --models "$cgs") ;;
  9)  a=(--part scenarios --bias trained --tag _cgtf_w_trained --models "$cgtf_w") ;;
  10) a=(--part scenarios --bias trained --tag _cgtf_s_trained --models "$cgtf_s") ;;
  11) a=(--part scenarios --bias trained --tag _lh_trained --models "$lh") ;;
  12) a=(--part scenarios --bias default --tag _cg_default --models "$cg") ;;
  13) a=(--part scenarios --bias default --tag _lh_default --models "$lh") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.joint_eval --config "$CONFIG" "${a[@]}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
