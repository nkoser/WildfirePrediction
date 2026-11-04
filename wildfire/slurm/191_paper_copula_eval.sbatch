#!/bin/bash
#SBATCH --job-name=wf_cop_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung der Frailty (copula_dep.py) mit theta aus 190, je STS-LH-Seed s = i % 3, Teil = i / 3:
#   0-2  cond       bedingte Vorhersage, Zeilen wie joint_cond 111 (alle Stichtage, kmax 70, dmax 4, 256 IS-Ziehungen):
#                   cop_cond, copg_cond                         -> cond_copula_s<s>.{json,npz}
#   3-5  shapes     Formen wie 110 (20.000 Stichtage), geschlossen: cop_cf, copg_cf -> shapes_copula_s<s>.*
#   6-8  scenarios  Szenarien wie 110 (32 Ziehungen) und Zahl brennender Pixel wie 118 (64 Ziehungen):
#                   cop_sim/_shuf, copg_sim/_shuf               -> scenarios_copula_s<s>.*, counts_copula_s<s>.*
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
s=$((i % 3)); part=(cond shapes scenarios); p=${part[$((i / 3))]}
CK=/data/Wildfire/checkpoints/paper
G=sts_cache_G_next_cube
"$PY" -m wildfire.paper.copula_dep --config "$CONFIG" --part "$p" --tag "_copula_s${s}" --seed-name "$s" \
    --lh-ckpt "$CK/${G}_abl_lh_s${s}_s${s}.pt"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
