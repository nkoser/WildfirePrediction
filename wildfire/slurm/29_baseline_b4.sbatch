#!/bin/bash
#SBATCH --job-name=wf_b4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/raid/data/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/raid/data/nik/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/vesselgpt_5
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data
#SBATCH --container-remap-root
#SBATCH --container-writable
# BEWUSST OHNE --container-name: pyxis verwendet einen gleichnamigen Container
# WIEDER und ignoriert dabei die --container-image-Zeile. Nach einem
# Imagewechsel lief Job 5847 dadurch weiter im alten Image (sr_env statt
# vmtk_2) und starb an einem numpy, das es dort gar nicht mehr geben sollte.
# Das Auspacken kostet wenige Sekunden - die Verwechslung kostete einen Tag.
# BEWUSST OHNE --partition und OHNE GPU: ein MLP auf 218 Merkmalen laeuft
# auf CPU in Minuten. GPU braucht erst B2 mit seinen Faltungen.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.models.mlp_survival --config "$CONFIG" --split "${SPLIT:-temporal}" --head "${HEAD:-bernstein}" ${EXTRA:-}
echo "[$(date -u +%FT%TZ)] exit $?"
