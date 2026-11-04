#!/bin/bash
#SBATCH --job-name=wf_bs_maps3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Black Swans: Risikokarten v3 = v2 zusaetzlich ohne Kalibrierung (wildfire/paper/black_swan_maps_v3.py), nur CPU.
# Liest set.npz/score_*.npz (175/176) und p40_black_swan_dense (177). Ausgabe NEU: figures/black_swan_v3${TAG}/
# (inkl. oe_by_event.csv), Manifest p40_black_swan_maps_v3${TAG}.
#   Rauchtest: TAG=_smoke EXTRA="--events 16 --models lh_ref --taus 30" sbatch 164_paper_black_swan_maps_v3.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.black_swan_maps_v3 --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
