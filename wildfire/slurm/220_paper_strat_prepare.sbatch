#!/bin/bash
#SBATCH --job-name=wf_strat_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Stratifizierter Split /firedata/MPCFire/cache_G_stratified als Cache-Variante (wildfire/paper/stratified_split.py):
# derived_data/paper/cache_G_stratified_L61/ NEU, split.npy aus der JSON, alles andere Symlinks auf cache_G_L61.
# Manifest p0_paper_cache_G_stratified_L61. Danach 221 (U-Net), 222 (STS), 223 (Tabelle).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.stratified_split --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
