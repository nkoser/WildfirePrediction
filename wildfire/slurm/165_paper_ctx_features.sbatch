#!/bin/bash
#SBATCH --job-name=wf_ctx_feat
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Zusatzeingaben fuer Extremlagen (wildfire/paper/ctx_features.py), nur CPU: Perzentil des Feuerwetters
# relativ zur eigenen Zelle 2000-2019, Kappungs-/NoData-Flags, regionale Feueraktivitaet der Saison.
# Ausgabe NEU: derived_data/${OUT:-ctx_features}/, Manifest p34_ctx_features[_OUT].
#   OUT=ctx_features_smoke EXTRA="--years 2019 2020 2021 2022 2023 2024 2025 --limit 2000"   Rauchtest
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.ctx_features --config "$CONFIG" --out-name "${OUT:-ctx_features}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
