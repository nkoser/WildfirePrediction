#!/bin/bash
#SBATCH --job-name=wf_hyb_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung der Hybride (hybrid_dep.py) mit theta aus 155, je Seed-Paar s = i % 3, Teil = i / 3:
#   0-2  cond       bedingte Vorhersage, Zeilen wie joint_cond 111 (alle Stichtage, kmax 70, dmax 4, 16 Pfade):
#                   hyb_cond (theta), hyb0_cond (theta = 0)          -> cond_hyb_s<s>.{json,npz}
#   3-5  shapes     Formen wie 110 (20.000 Stichtage, 32 Pfade): hyb_mc -> shapes_hyb_s<s>.{json,npz}
#   6-8  scenarios  Szenarien wie 110 (32 Pfade) und Zahl brennender Pixel wie 118 (64 Ziehungen):
#                   hyb_sim, hyb_shuf -> scenarios_hyb_s<s>.*, counts_hyb_s<s>.*
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
s=$((i % 3)); part=(cond shapes scenarios); p=${part[$((i / 3))]}
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
"$PY" -m wildfire.paper.hybrid_dep --config "$CONFIG" --part "$p" --tag "_hyb_s${s}" --seed-name "$s" \
    --lh-ckpt "$CK/${G}_abl_lh_s${s}_s${s}.pt" --cg-ckpt "$CK/${G}_abl_cg_scratch_s${s}_s${s}.pt"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
