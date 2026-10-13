#!/bin/bash
# BRAUCHT DIE ROHDATEN - im Container unter /firedata (Bind in _apptainer.sh).
#SBATCH --job-name=wf_audit_caches
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Stufe 4: HDF5-Fenster-Caches und Sample-Indizes der vier Varianten.
# Struktur, Kanalzahl, Fensterlaenge, Splits, Klassenverteilung,
# NaN-Anteile und Kanal-Statistiken auf einer Zufallsstichprobe.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
"$PY" -m wildfire.audit.s4_caches --config "$CONFIG"
echo "[$(date -u +%FT%TZ)] exit $?"
