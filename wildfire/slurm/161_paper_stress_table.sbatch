#!/bin/bash
#SBATCH --job-name=wf_stress_tab
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Stresstest, Schritt 2 (nur CPU): Ensemble und Seeds aus derived_data/stress_test/*${TAG}.npz
# zusammenfassen -> Manifest p32_stress_test${TAG} (stress_test.py --summarize), dann Markdown-Tabellen
# und CSV unter metrics/stress_test/ (stress_table.py). Nach 160:
#   sbatch --dependency=afterok:<160-Job> 161_paper_stress_table.sbatch
#   Rauchtest: TAG=_smoke MODELS="lh" EXTRA="--limit 256" sbatch ...
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.stress_test --config "$CONFIG" --summarize --models ${MODELS:-lh cg_scratch} \
    --tag "${TAG:-}" ${EXTRA:-} || { rc=$?; echo "exit $rc"; exit $rc; }
"$PY" -m wildfire.paper.stress_table --config "$CONFIG" --tag "${TAG:-}"
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
