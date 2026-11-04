#!/bin/bash
#SBATCH --job-name=wf_dayca
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Tag-fuer-Tag-Ausbreitung: Transformer ueber 30 Kontexttage setzt je Tag die Parameter eines CA-Schritts
# (wildfire/paper/fire_dayca.py), von Grund auf; Ereignisse wie symbolic_spread / firemap_bench.
# Ausgabe NEU: metrics/fire_dayca${TAG}/.  Ohne GPU (Smoke): sbatch --gres=none ... mit EXTRA="--limit 2 --epochs 1 --paths 4".
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
"$PY" -m wildfire.paper.fire_dayca --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
