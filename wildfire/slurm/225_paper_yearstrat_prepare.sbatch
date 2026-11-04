#!/bin/bash
#SBATCH --job-name=wf_ys_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Jahres-stratifizierter Split year_strat (aus 224/split_audit) als Cache-Variante cache_G_yearstrat:
# stratified_split.py mit der Overlay-Config audit_splitsv2.yaml (dataset_root = $EXP/dataset_overlay/MPCFire,
# cache_G_yearstrat/samples_variant_G.json -> derived_data/splits_v2/year_strat). Ausgabe NEU:
# derived_data/paper/cache_G_yearstrat_L61/ (split.npy, Rest Symlinks auf cache_G_L61). Danach 226, 227, 228.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.stratified_split --config "$CONFIG" --cache cache_G_yearstrat ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
