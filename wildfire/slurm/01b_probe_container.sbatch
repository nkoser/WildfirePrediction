#!/bin/bash
#SBATCH --job-name=wf_probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Reine Sondierung: Sind im Container die drei Binds da, liest /firedata die
# Rohdaten, ist /firedata wirklich read-only, und sind alle Pakete importierbar?
# Sourct BEWUSST NICHT _common.sh, damit dieser Job auch dann Auskunft gibt,
# wenn die Interpreterpruefung dort scheitern wuerde.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
echo "=== Host und Job ==="
hostname; echo "job=${SLURM_JOB_ID:-?} image=${APPTAINER_CONTAINER:-?}"; date -u +%FT%TZ

echo; echo "=== Binds ==="
for d in /workspace/WildfirePrediction /data/Wildfire /firedata/MPCFire/Entire_Canada_Maps; do
    if [ -d "$d" ]; then echo "  OK    $d  ($(ls "$d" 2>/dev/null | wc -l) Eintraege)"
    else echo "  FEHLT $d"; fi
done
if touch /firedata/.wf_write_probe 2>/dev/null; then
    rm -f /firedata/.wf_write_probe; echo "  WARNUNG: /firedata ist beschreibbar"
else
    echo "  OK    /firedata ist read-only"
fi
echo "  tar-Archive: $(find /firedata/MPCFire/Entire_Canada_Maps -name '*.tar' | wc -l) (erwartet 338)"

echo; echo "=== Pakete ==="
python - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
for m in ("numpy", "scipy", "torch", "rasterio", "pyproj", "h5py", "sklearn", "matplotlib", "pytest", "yaml"):
    try:
        mod = importlib.import_module(m); print(f"  OK    {m:11s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  FEHLT {m:11s} {e}")
import torch
print("  cuda verfuegbar:", torch.cuda.is_available())
PY
