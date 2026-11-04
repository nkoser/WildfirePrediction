#!/bin/bash
#SBATCH --job-name=wf_fwif_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Ungekappte FWI-Werte, Schritt 1 (wildfire/paper/fwi_float.py --mode probe), nur lesen, nur CPU:
# Datentyp / Skalierung der Roh-GeoTIFFs FWI, ERA5, VPD; Skalen der FWI-Baender aus den CFFDRS-
# Gleichungen BUI = f(DMC, DC) und FWI = g(ISI, BUI); DC-Zuwachs gegen Temperatur.
# Ausgabe NEU: Manifest p33_fwi_float_probe. MODE=probe2: Skalen mit Offset, rueckrechenbare Anteile
# gekappter Zelltage, DC-Zuwachs je Monat (Manifest p33_fwi_float_probe2).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" -m wildfire.paper.fwi_float --config "$CONFIG" --mode "${MODE:-probe}" ${EXTRA:-}
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
