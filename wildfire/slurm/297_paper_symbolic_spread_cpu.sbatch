#!/bin/bash
#SBATCH --job-name=wf_sym_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Symbolische Ausbreitung (wildfire/paper/symbolic_spread.py), CPU-Schritte -> metrics/symbolic_spread${TAG}/
#   STAGE=events  Trainings-Ereignisse (Jahre des Trainingssplits, Regel wie firemap_bench)
#   STAGE=smoke   alle Modelle, 3 Ereignisse je Split, 1 Epoche, 10 Pfade -> metrics/symbolic_spread_smoke/
#   STAGE=report  summary.md (mit den firemap_bench-Laeufen)
#   STAGE=maps    beobachtete Masken + Karten aller Laeufe fuer den Viewer (maps_viewer/data_<report-name>.json)
# GPU-Anpassung: 298_paper_symbolic_spread_fit.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
case ${STAGE:?STAGE=events|smoke|report} in
  smoke)
    mkdir -p $EXP/metrics/symbolic_spread_smoke
    mkdir -p $EXP/metrics/symbolic_spread_smoke${SMOKE_RUN:-}; cp --update=none $EXP/metrics/symbolic_spread${TAG:-}/events_train.json $EXP/metrics/symbolic_spread_smoke${SMOKE_RUN:-}/
    rc=0
    for m in ${SMOKE_MODELS:-ca_lin ca_full rd rdca}; do
      "$PY" -m wildfire.paper.symbolic_spread fit --config "$CONFIG" --tag _smoke${SMOKE_RUN:-} --model $m --limit 3 --epochs ${SMOKE_EPOCHS:-1} --paths 10 ${EXTRA:-} || rc=1
    done ;;
  *)
    "$PY" -m wildfire.paper.symbolic_spread "$STAGE" --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-}; rc=$? ;;
esac
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
