#!/bin/bash
#SBATCH --job-name=wf_b2_unet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_b2_unet
# ---------------------------------------------------------------------------
# ERSTER Job im Projekt, der eine GPU braucht - deshalb hier --gres=gpu:1.
# Alle vorherigen Schritte waren CPU-gebunden und liefen ohne.
#
# Weiterhin OHNE --partition: das hatte einen CPU-Job stundenlang in der
# GPU-Queue haengen lassen. Fuer einen GPU-Job ist die Default-Partition
# ohnehin die richtige.
#
# 8 CPUs fuer die Dataloader: Der Engpass ist das Lesen der Patches aus dem
# 340-GB-Wuerfel, nicht die GPU. Gemessen rund 260 Samples/s je Leser.
#
# Vor dem echten Lauf IMMER den Smoke-Test:
#   SMOKE=1 sbatch wildfire/slurm/28_train_b2.sbatch
# ---------------------------------------------------------------------------
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PYTORCH_ALLOC_CONF=expandable_segments:True
"$PY" -m wildfire.train --config "$CONFIG" \
    --model-config "${MODEL_CONFIG:-/workspace/WildfirePrediction/wildfire/configs/b2_unet.yaml}" \
    ${SMOKE:+--smoke}
echo "[$(date -u +%FT%TZ)] exit $?"
