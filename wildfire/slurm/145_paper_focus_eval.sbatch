#!/bin/bash
#SBATCH --job-name=wf_focus_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Paper-Protokoll je dt auf allen Test-Stichproben und auf neuen Zuendungen ign30 (wildfire/paper/focus_eval.py):
# Modelle aus 153 (sts_inputs), 159 (sts_focus), Referenzen, logistische Basis, U-Net-Nachbau.
# Ausgaben NEU: metrics/focus_eval${TAG}/, Manifest p45_focus_eval${TAG}.
#   Rauchtest: TAG=_smoke MODELS="lh_ref lh_resid_smoke:focus=resid" EXTRA="--seeds 0 --limit 500"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.focus_eval --config "$CONFIG" --tag "${TAG:-}" \
    --models ${MODELS:-lh_ref cg_ref lh_fireonly:inputs=fireonly lh_nofire:inputs=nofire cg_fireonly:inputs=fireonly cg_nofire:inputs=nofire lh_fdrop:focus=fdrop cg_fdrop:focus=fdrop lh_resid:focus=resid lh_ignw:focus=ignw cg_ignw:focus=ignw lh_fdrop_mono:focus=fdrop} ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
