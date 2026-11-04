#!/bin/bash
#SBATCH --job-name=wf_split_abl
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Uebersicht der Varianten-Ablation auf drei Splits (wildfire/paper/split_ablation_table.py), nur aus Ergebnissen
# von 145/168 (zeitlich), 235/236 (neue Splits) und den U-Net-Vorhersagen von 226/230.
# Ausgaben NEU: metrics/split_ablation${TAG}/, Manifest p51_split_ablation${TAG}.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.split_ablation_table --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
