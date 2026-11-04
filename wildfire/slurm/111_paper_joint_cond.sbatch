#!/bin/bash
#SBATCH --job-name=wf_joint_cond
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Bedingte Vorhersage (paper/joint_cond.py --part run): alle Test-Stichtage mit erstem Feuer im Ausschnitt an
# Tag 1..70, Pixel bis Abstand 4 zum Ausloeser, 16 Pfade; nachkalibrierte Modelle (109).
#   0  Contagion warm (cond, quiet)    1  Contagion von Grund auf (cond, quiet)    2  STS-LH
# Ausgabe NEU: metrics/paper_joint/cond_<name>.{json,npz}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
case $i in
  0) a=(--tag _cg --models "cg=cg:$CK/${G}_abl_cg_s?_s?.pt") ;;
  1) a=(--tag _cgs --models "cgs=cg:$CK/${G}_abl_cg_scratch_s?_s?.pt") ;;
  2) a=(--tag _lh --models "lh=lh:$CK/${G}_abl_lh_s?_s?.pt") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.joint_cond --config "$CONFIG" --part run --calib-dir calib "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
