#!/bin/bash
#SBATCH --job-name=wf_cube_array
#SBATCH --array=0-25%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/data/Wildfire/logs/%x_%A_%a.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
# ---------------------------------------------------------------------------
# Ein Array-Task JE JAHR, Index 0..25 -> Jahr 2000..2025.
#
# Die Array-ID ist NICHT die Jahreszahl: Slurm begrenzt Array-Indizes auf
# MaxArraySize (Standard 1001), 2025 waere ungueltig. Der Offset steht in
# BASE_YEAR und kann ueberschrieben werden.
#
# Ressourcen bewusst klein: 8 CPUs / 64 GB entsprechen den Anforderungen,
# die in diesem Cluster nachweislich durchlaufen (siehe die uebrigen Repos
# unter /workspace). Groessere Anforderungen bleiben in der Default-Partition
# hinter GPU-Jobs haengen.
#
# Der Schritt ist IO- und nicht CPU-gebunden: 13 Archive je Jahr werden
# sequentiell gelesen. Acht Worker reichen; %2 begrenzt auf zwei
# gleichzeitige Tasks, damit die Platte nicht saettigt.
#
# Einzelne Jahre nachziehen (Index, nicht Jahreszahl!):
#   2017 -> Index 17, 2019 -> Index 19
#   sbatch --array=17,19 wildfire/slurm/15b_build_cube_array.sbatch
#
# Kein --container-name: bei Array-Jobs wuerden sich sonst alle Tasks
# denselben Container teilen wollen.
# ---------------------------------------------------------------------------
# BEWUSST OHNE --partition und ohne --gres: reine CPU-Jobs nehmen so die
# Default-Partition und warten nicht in der GPU-Queue. Ein explizites
# --partition=gpu liess Job 5196 ueber zwei Stunden in PENDING haengen,
# obwohl der Knoten 224 CPUs hat. Gleiches Muster wie in AneuG und
# femur-sr-paper.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
BASE_YEAR="${BASE_YEAR:-2000}"
IDX="${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich (sbatch --array=...)}"
YEAR=$(( BASE_YEAR + IDX ))
echo "[array] Task $IDX -> Jahr $YEAR"
"$PY" -m wildfire.data.cube --config "$CONFIG" --years "$YEAR" --workers "${WORKERS:-8}"
echo "[$(date -u +%FT%TZ)] exit $?"
