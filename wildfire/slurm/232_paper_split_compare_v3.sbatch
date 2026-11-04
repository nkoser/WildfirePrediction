#!/bin/bash
#SBATCH --job-name=wf_split_cmp3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Tabelle ueber vier Splits: zeitlich, zufaellig stratifiziert, Jahre (cache_G_yearstrat), Bloecke (cache_G_blockstrat),
# wildfire/paper/split_compare_v3.py, nur aus Manifesten von 221/222/226/227/230/231. Ausgaben NEU:
# metrics/split_compare_v3${TAG}/summary.md, Manifest p49_split_compare_v3${TAG}.
#   sbatch --dependency=afterok:<230>:<231> 232_paper_split_compare_v3.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.split_compare_v3 --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
