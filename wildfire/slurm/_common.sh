# Wird von allen sbatch-Skripten gesourced - IM Container (siehe _apptainer.sh).
# Haelt Umgebung, Pfade und Logging an einer Stelle.
#
# Alle Pfade hier sind Container-Pfade. Die Binds in _apptainer.sh bilden sie
# auf den Host ab (/data/Wildfire -> /scratch/shared/nik/Wildfire usw.).
set -uo pipefail

REPO="/workspace/WildfirePrediction"
EXP="/data/Wildfire"
CONFIG="${CONFIG:-$REPO/wildfire/configs/audit.yaml}"

# Interpreter: das Python des Images (wildfire.sif, Basis nvcr pytorch). Alle
# Pakete sind dort installiert; das fruehere $EXP/pydeps-<cpXY> entfaellt.
#   PY=/pfad/zu/python sbatch <skript>    uebersteuert die Wahl
PY="${PY:-$(command -v python3 || command -v python)}"
if [ -z "$PY" ] || ! "$PY" -c "import numpy" >/dev/null 2>&1; then
    echo "ABBRUCH: kein Interpreter mit numpy gefunden (PY=$PY)." >&2
    echo "  Laeuft das Skript ueberhaupt im Container? APPTAINER_CONTAINER=${APPTAINER_CONTAINER:-<leer>}" >&2
    exit 2
fi
PYTAG=$("$PY" -c "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$EXP/logs" "$EXP/data_manifests" "$EXP/reports" "$EXP/figures" \
         "$EXP/checkpoints" "$EXP/predictions" "$EXP/metrics"

echo "[$(date -u +%FT%TZ)] host=$(hostname) job=${SLURM_JOB_ID:-interactive} py=$PY ($PYTAG)"
echo "[cfg] $CONFIG   image=${APPTAINER_CONTAINER:-<kein Container>}"
# Nur berichten, nicht abbrechen: welcher Job welches Paket braucht, weiss er selbst.
"$PY" - <<'PYCHK'
import importlib.util
missing = [m for m in ("scipy", "rasterio", "h5py", "torch", "sklearn", "matplotlib", "pytest", "yaml")
           if importlib.util.find_spec(m) is None]
print("[env] fehlende optionale Pakete: " + (", ".join(missing) if missing else "keine"))
PYCHK
