#!/bin/bash
#SBATCH --job-name=wf_predict_map
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Risikokarte fuer einen Kartenausschnitt zu beliebigen Zeitpunkten.
#
# BEWUSST OHNE --gres: die Inferenz ist ein einzelner Vorwaertsdurchlauf je
# Kachel und laeuft auf CPU in Minuten. Wer eine GPU will, haengt sie an:
#   sbatch --gres=gpu:1 wildfire/slurm/31_predict_map.sbatch
# Das Skript nimmt sie dann automatisch (torch.cuda.is_available).
#
# Parameter ueber Umgebungsvariablen:
#   CKPT   Pfad zum Checkpoint            (Pflicht)
#   DATE   Stichtag, z. B. 2023-07-15     (Pflicht)
#   TAUS   Horizonte in Tagen             (Standard 1,7,14,30,60,100)
#   BBOX   row0,row1,col0,col1            (Standard: ganzes Gitter)
#   TILE   Kachelgroesse                  (Standard 256)
#   HALO   Ueberlappung fuer den Rand     (Standard 16)
#
# Beispiel:
#   CKPT=/data/Wildfire/checkpoints/b2_unet_temporal/best.pt \
#   DATE=2023-07-15 TAUS=1,7,30,100 BBOX=400,700,300,800 \
#   sbatch wildfire/slurm/31_predict_map.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
: "${CKPT:?CKPT muss gesetzt sein - Pfad zum Modell-Checkpoint}"
: "${DATE:?DATE muss gesetzt sein - Stichtag im Format JJJJ-MM-TT}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "[map] keine GPU, laeuft auf CPU"
"$PY" -m wildfire.predict_map --config "$CONFIG" \
    --checkpoint "$CKPT" --date "$DATE" \
    --taus "${TAUS:-1,7,14,30,60,100}" \
    --tile "${TILE:-256}" --halo "${HALO:-16}" \
    ${BBOX:+--bbox "$BBOX"} ${OUT:+--out "$OUT"}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
