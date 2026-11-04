#!/bin/bash
#SBATCH --job-name=wf_sym_fitc
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# CPU-Variante von 298 (GPUs belegt; die Anpassung braucht nur Minuten). Symbolische Ausbreitung anpassen und auf den firemap_bench-Ereignissen (val/test) auswerten, ohne Warmstart.
# Array: Modell = MODELS[i]  (0 ca_lin, 1 ca_coop, 2 ca_age, 3 ca_size, 4 ca_full, 5 rd, 6 rdca, 7 rd_ens, 8 rd_ens_fss, 9 rd_fss, 10 ca_dir, 11 ca_classic, 12 rd_classic, 13 rd_ens_classic, 14 ca_wx, 15 ca_wx3, 16 ca_wx7) -> metrics/symbolic_spread${TAG}/
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
MODELS=(ca_lin ca_coop ca_age ca_size ca_full rd rdca rd_ens rd_ens_fss rd_fss ca_dir ca_classic rd_classic rd_ens_classic ca_wx ca_wx3 ca_wx7)
export OMP_NUM_THREADS=16
"$PY" -m wildfire.paper.symbolic_spread fit --config "$CONFIG" --tag "${TAG:-}" --model ${MODELS[${SLURM_ARRAY_TASK_ID:?}]} ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
