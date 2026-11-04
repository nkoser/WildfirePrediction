#!/bin/bash
#SBATCH --job-name=wf_bs_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Block-stratifizierter Split block_strat (aus 224/split_audit) als Cache-Variante cache_G_blockstrat:
# splits_v2_prepare.py (liest split.npy direkt, gepurgte Zeilen = "drop"), Overlay-Config audit_splitsv2.yaml
# (dataset_root = $EXP/dataset_overlay/MPCFire, cache_G_blockstrat/norm_stats.npz -> cache_G). Ausgabe NEU:
# derived_data/paper/cache_G_blockstrat_L61/ (split.npy, Rest Symlinks auf cache_G_L61). Danach 230, 231, 232.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.splits_v2_prepare --config "$CONFIG" --name block_strat --cache cache_G_blockstrat ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
