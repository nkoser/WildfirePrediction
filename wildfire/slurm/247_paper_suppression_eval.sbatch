#!/bin/bash
#SBATCH --job-name=wf_suppression_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Brandbekaempfung als Stoerfaktor der Infrastruktur-Variablen, nur CPU (wildfire/paper/suppression_eval.py).
# Teil A: NFDB RESPONSE / ATTK_DATE / OUT_DATE gegen Bevoelkerung am Zuendort.
# Teil B: Kalibrierung der Gruppen-Dropout-Netze (Scores aus 218, metrics/shapley_groups${SRC_TAG}) je
#         Bevoelkerungs- und Bekaempfungsklasse.
#   sbatch 247_paper_suppression_eval.sbatch
#   Rauchtest: TAG=_smoke sbatch 247_...
# Ausgaben NEU: metrics/suppression_eval${TAG}/{summary.md,fires.csv,model_groups.csv}, Manifest p55_suppression_eval${TAG}
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.suppression_eval --config "$CONFIG" --src-tag "${SRC_TAG:-}" --tag "${TAG:-}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
