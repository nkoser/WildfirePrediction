#!/bin/bash
#SBATCH --job-name=wf_paper_ours
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Paper-Protokoll, Schritt 5: unser B4 (NFDB-trainiert) auf den Stichproben des
# Papers, Verschiebe-Protokoll dt = 1/7/15/30, Vergleich mit dem 3D-U-Net auf
# derselben Teilmenge. Braucht den Wuerfel, 29_baseline_b4 (Checkpoint),
# 40_paper_prepare (T = 30) und 42_paper_reconstruct (_paperdt, Vorhersagen).
#   0 -> cache_A   1 -> cache_G   2 -> cache_H   3 -> cache_Y
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CACHES=(cache_A cache_G cache_H cache_Y)
"$PY" -m wildfire.paper.ours_on_paper --config "$CONFIG" \
    --cache "${CACHES[${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}]}" --head "${HEAD:-discrete}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
