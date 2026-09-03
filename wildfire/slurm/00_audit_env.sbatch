#!/bin/bash
#SBATCH --job-name=wf_audit_env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_audit_env
# ---------------------------------------------------------------------------
# CPU-only: keine --gres/--partition, laeuft in der Default-Partition.
# Legt die Geo-Abhaengigkeiten des Audits persistent unter $DEPS ab, damit
# die folgenden Audit-Jobs nicht jedes Mal neu installieren muessen.
# Der Rohdatenmount ist bewusst :ro -- Read-only wird auf Mount-Ebene erzwungen.
# ---------------------------------------------------------------------------
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh

mkdir -p "$DEPS"
"$PY" -m pip install -q --no-input --target "$DEPS" --upgrade \
    rasterio pyproj affine

echo "=== Verifikation ==="
"$PY" - <<'EOF'
import importlib, sys
for m in ("numpy", "pandas", "h5py", "tifffile", "rasterio", "pyproj", "matplotlib", "scipy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?'):12s} {getattr(mod, '__file__', '')}")
    except Exception as e:
        print(f"  {m:12s} FEHLT: {type(e).__name__}: {e}")
        sys.exit(1)
EOF
echo "[$(date -u +%FT%TZ)] exit $?"
