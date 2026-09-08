#!/bin/bash
# BRAUCHT DIE ROHDATEN. Auf dem Host liegen sie unter /cold/Niklas/FireData;
# in dieser interaktiven Session sind sie als /firedata sichtbar. Fuer einen
# pyxis-Container muss die HOST-Quelle stehen - ein Mount von /firedata
# scheiterte, weil es diesen Pfad auf dem Host nicht gibt.
#
# Ziel ist /mnt und nicht /firedata: Enroot legt fehlende Mountpunkte im Image
# nicht an, und ob das SR-Image ein /firedata hat, ist ungeprueft. /mnt gibt es
# in jedem Image. Passend dazu setzt WF_DATASET_ROOT den Rohdatenpfad um; der
# Wert landet ueber write_manifest in der Provenienz.
#SBATCH --job-name=wf_fire_index
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/data/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/raid/data/nik/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/vesselgpt_5
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/cold/Niklas/FireData:/mnt:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
# BEWUSST OHNE --container-name: pyxis verwendet einen gleichnamigen Container
# WIEDER und ignoriert dabei die --container-image-Zeile. Nach einem
# Imagewechsel lief Job 5847 dadurch weiter im alten Image (sr_env statt
# vmtk_2) und starb an einem numpy, das es dort gar nicht mehr geben sollte.
# Das Auspacken kostet wenige Sekunden - die Verwechslung kostete einen Tag.
# Duenner Index aller Feuertage und Zuendungen je Pixel (CSR).
# Grundlage der Survival-Labels: "naechste Zuendung nach t" per binaerer
# Suche, und "brennt gerade" fuer den Landmark-Ausschluss.
export WF_DATASET_ROOT=/mnt/MPCFire   # Rohdaten liegen im Container unter /mnt
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.data.fire_index --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
