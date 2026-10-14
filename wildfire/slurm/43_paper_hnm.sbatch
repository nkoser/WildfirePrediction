#!/bin/bash
#SBATCH --job-name=wf_paper_hnm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Paper-Protokoll, Schritt 4: FWI-HNM-Gefahrenwert s nachbauen (P99-Normierung,
# Gewichte aus standardisierter Mittelwertdifferenz) und s fuer alle Stichproben
# der vier Caches messen. Braucht den Wuerfel und 40_paper_prepare (T = 30).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.hnm --config "$CONFIG" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
