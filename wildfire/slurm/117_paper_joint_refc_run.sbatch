#!/bin/bash
#SBATCH --job-name=wf_refc_run
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Neustart an Tag k (joint_refc --part run), Zeilen wie 111, nachkalibrierte Modelle (109/114), 16 Pfade.
#   0  Contagion warm (cg_refc)    1  Contagion von Grund auf (cgs_refc)
#   2  STS-LH (lh_refc) und die Basismodelle lh_adj (je Seed auf Val gefittet) und emp (Faustregel)
# Ausgabe NEU: metrics/paper_joint/refc_<name>.{json,npz}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
case $i in
  0) a=(--tag _cg --models "cg=cg:$CK/${G}_abl_cg_s?_s?.pt") ;;
  1) a=(--tag _cgs --models "cgs=cg:$CK/${G}_abl_cg_scratch_s?_s?.pt") ;;
  2) a=(--tag _lh --with-baselines --models "lh=lh:$CK/${G}_abl_lh_s?_s?.pt") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.joint_refc --config "$CONFIG" --part run --build-name test --calib-dir calib "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
