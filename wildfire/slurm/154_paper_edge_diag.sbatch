#!/bin/bash
#SBATCH --job-name=wf_edge_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Randdiagnose (wildfire/paper/edge_diag.py): O/E-Karte je Position und Verschiebungstest des Pixelkopfs.
# Standard: lh_ref cg_ref lh_mono cg_mono lhbce10 utae_bce10 convlstm_bce10 (U-TAE/ConvLSTM: Reflect-Padding).
# Ausgaben NEU: metrics/edge_diag${TAG}/, figures/edge_diag${TAG}/, Manifest p42_edge_diag${TAG}.
#   Rauchtest: TAG=_smoke EXTRA="--n-oe 300 --n-shift 6 --models lh_ref cg_ref utae_bce10" sbatch 154_paper_edge_diag.sbatch
#   Reflect-Laeufe aus 153: EXTRA="--models lh_ref lh_reflect:all:reflect cg_ref cg_reflect:all:reflect" TAG=_reflect
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.edge_diag --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
