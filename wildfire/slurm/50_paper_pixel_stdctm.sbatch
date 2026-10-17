#!/bin/bash
#SBATCH --job-name=wf_paper_pxst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# ST-DCTM pro Pixel: Survival-Kurve fuer jedes der 13x13 Pixel, Ereignis =
# erstes Feuer dieses Pixels nach dem Stichtag (Cache-Feuermaske), zensiert
# am Zieltag; Auswertung im dt-Protokoll wie 49. Braucht 40_paper_prepare (_L61).
#   CACHE=cache_G (Standard)   EXTRA="--seeds 0 --epochs 10"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.pixel_stdctm --config "$CONFIG" --cache "${CACHE:-cache_G}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
