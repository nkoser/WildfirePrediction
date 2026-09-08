#!/bin/bash
#SBATCH --job-name=wf_probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/raid/data/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/raid/data/nik/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/vesselgpt_5
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/cold/Niklas/FireData:/mnt:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
# ---------------------------------------------------------------------------
# Reine Sondierung. Sourct BEWUSST NICHT _common.sh, damit dieser Job auch
# dann Auskunft gibt, wenn die Interpretersuche scheitert - genau das war der
# Fehler bei Job 5845. Kein Rechenaufwand, nur Bestandsaufnahme.
# ---------------------------------------------------------------------------
echo "=== Host und Job ==="
hostname; echo "job=${SLURM_JOB_ID:-?}"; date -u +%FT%TZ

echo; echo "=== Mounts: sind die drei Pfade da? ==="
for d in /workspace/WildfirePrediction /data/Wildfire /mnt; do
    if [ -d "$d" ]; then echo "  OK    $d  ($(ls "$d" 2>/dev/null | wc -l) Eintraege)"
    else echo "  FEHLT $d"; fi
done
echo "  Inhalt von /mnt:"; ls /mnt 2>&1 | head -5

echo; echo "=== conda ==="
ls -d /opt/miniconda3 /miniconda3 /opt/conda 2>/dev/null || echo "  keines der ueblichen Verzeichnisse"
for c in /opt/miniconda3/bin/conda /miniconda3/bin/conda /opt/conda/bin/conda; do
    [ -x "$c" ] && { echo "  gefunden: $c"; "$c" env list; break; }
done

echo; echo "=== Interpreter im PATH ==="
command -v python python3 pip 2>/dev/null || echo "  keiner"

echo; echo "=== alle python-Binaries unter /opt und / ==="
find /opt /usr/bin /usr/local/bin -maxdepth 4 -name "python3*" -type f 2>/dev/null | head -12

echo; echo "=== Pakettest je Kandidat ==="
for p in $(find /opt /usr/bin /usr/local/bin -maxdepth 5 -name "python3*" -type f 2>/dev/null | head -12); do
    out=$("$p" -c "import numpy,h5py,scipy;print('numpy',numpy.__version__)" 2>&1 | tail -1)
    echo "  $p  ->  $out"
done

echo; echo "=== torch vorhanden? ==="
for p in $(find /opt -maxdepth 5 -name "python3*" -type f 2>/dev/null | head -6); do
    echo "  $p -> $("$p" -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"
done
echo; echo "fertig"
