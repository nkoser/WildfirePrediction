#!/bin/bash
#SBATCH --job-name=wf_b2_unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# ERSTER Job im Projekt, der eine GPU braucht - deshalb hier --gres=gpu:1.
# Alle vorherigen Schritte waren CPU-gebunden und liefen ohne.
#
# MIT --partition=gpu. Frueher stand hier das Gegenteil, mit der Begruendung,
# ein CPU-Job habe in der GPU-Queue gehangen. Das war eine falsche
# Verallgemeinerung: Der Hinweis gilt fuer CPU-Jobs. Dieser Job fordert als
# einziger im Projekt --gres=gpu:1 an, und wenn die Default-Partition keine
# GPUs hat, ist diese Anforderung dort NIE erfuellbar - der Job pendet
# unbegrenzt, ohne Fehlermeldung. Genau das ist passiert.
#
# Heisst die Partition anders, ohne Datei-Aenderung uebersteuerbar:
#   sbatch --partition=<name> wildfire/slurm/28_train_b2.sbatch
# Namen zeigt:  sinfo -o '%P %G %a %D'
#
# 32 CPUs und 400 GB. Beides war vorher zu klein angesetzt (8 / 96 GB) und
# hat Rechenzeit verschenkt - der Knoten hat 224 CPUs und 2015 GB.
#
#   CPUs   begrenzen die Zahl der Dataloader-Prozesse. Mit 8 lag der Durchsatz
#          bei 8 x 106 = 850 Beispiele/s, waehrend die B200 unter 5 GB ihrer
#          183 GB belegte und ueberwiegend auf Daten wartete.
#
#   RAM    ist der eigentliche Hebel. Die Arbeitsmenge einer Epoche sind
#          253 GB (18 Trainingsjahre a 14 GB). Unter cgroups zaehlt der
#          Seitencache gegen --mem, mit 96 GB passten also nur 38 % hinein und
#          jede Epoche las den Rest erneut von der Platte. Mit 400 GB liegt der
#          gesamte Trainingswuerfel nach der ersten Epoche im Cache.
#
# NICHT geaendert wurden Batchgroesse und Netzbreite. Beide wuerden die
# Optimierung bzw. die Modellkapazitaet veraendern und damit den Vergleich mit
# B4 verfaelschen - hier geht es nur um Infrastruktur, nicht um das Experiment.
#
# Vor dem echten Lauf IMMER den Smoke-Test:
#   SMOKE=1 sbatch wildfire/slurm/28_train_b2.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PYTORCH_ALLOC_CONF=expandable_segments:True
"$PY" -m wildfire.train --config "$CONFIG" \
    --model-config "${MODEL_CONFIG:-/workspace/WildfirePrediction/wildfire/configs/b2_unet.yaml}" \
    ${SMOKE:+--smoke}
echo "[$(date -u +%FT%TZ)] exit $?"
