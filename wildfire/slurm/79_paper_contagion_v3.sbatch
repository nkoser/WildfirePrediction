#!/bin/bash
#SBATCH --job-name=wf_contagion3
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS Version 3: wie Version 2 (65, von Grund auf), aber der Rumpf
# sieht die Feuerkanaele nicht. Feuer in der Naehe wirkt nur ueber b_j und den
# Kern, die Zuendung kann die Ausbreitung nicht uebernehmen.
#   0-2  --cg-blind thermal (Feuermaske, Kanaele 45-53)  -> ..._d128_cg3t_s<seed>
#   3-5  --cg-blind modis   (Feuermaske, Kanaele 38-53)  -> ..._d128_cg3m_s<seed>
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3))
if [ $i -lt 3 ]; then blind=thermal; tag=cg3t; else blind=modis; tag=cg3m; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" --train-source cube \
    --target next_patch --d 128 --depth 6 --epochs 10 --steps-per-epoch 3000 --seeds $seed --save-ckpt \
    --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve --cg-blind $blind \
    --tag "_d128_${tag}_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
