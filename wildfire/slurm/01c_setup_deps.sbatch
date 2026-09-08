#!/bin/bash
#SBATCH --job-name=wf_setup_deps
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/data/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/raid/data/nik/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/vesselgpt_5
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data
#SBATCH --container-remap-root
#SBATCH --container-writable
# ---------------------------------------------------------------------------
# NORMALERWEISE NICHT NOETIG. Das Image vesselgpt_5 bringt alles mit, was das
# Projekt braucht; rasterio liegt in /data/Wildfire/pydeps-cp311. Dieser Job
# ist der Reparaturweg, falls ein Image gewechselt wird oder Pakete fehlen -
# er ergaenzt ein Environment um die fehlenden Pakete, EINMAL, in ein
# persistentes Verzeichnis auf /data.
# EINMAL, in ein persistentes Verzeichnis auf /data, damit kein Rechenjob
# spaeter pip aufrufen muss (und schon gar kein GPU-Job).
#
# Befund aus Sondierungsjob 5846: sr_env (Python 3.12) hat numpy und
# torch 2.11.0+cu128, aber kein h5py. Das bestehende /data/Wildfire/pydeps
# stammt aus dieser interaktiven Session und ist fuer cpython-311 kompiliert -
# unter 3.12 nicht importierbar. Deshalb ein eigener Baum je Python-Version.
#
# CPU-Job, bewusst ohne --gres und ohne --partition.
#
#   sbatch wildfire/slurm/01c_setup_deps.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail
EXP=/data/Wildfire

PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in /opt/miniconda3/envs/sr_env/bin/python3.12 \
             /opt/miniconda3/envs/sr_env/bin/python; do
        [ -x "$c" ] && { PY="$c"; break; }
    done
fi
[ -x "${PY:-}" ] || { echo "ABBRUCH: sr_env-Python nicht gefunden" >&2; exit 2; }

PYTAG=$("$PY" -c "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
DEPS="${DEPS:-$EXP/pydeps-$PYTAG}"
mkdir -p "$DEPS" "$EXP/logs"
echo "[env] $PY"
"$PY" -c "import sys;print('[env] Python', sys.version.split()[0])"
echo "[env] Ziel: $DEPS"

export PYTHONPATH="$DEPS:/workspace/WildfirePrediction:${PYTHONPATH:-}"

# Modulname -> pip-Name, wo sie sich unterscheiden
pipname () {
    case "$1" in
        sklearn) echo "scikit-learn" ;;
        *)       echo "$1" ;;
    esac
}

echo
echo "=== Bestandsaufnahme vor der Installation ==="
for m in numpy torch scipy rasterio h5py sklearn matplotlib pytest; do
    v=$("$PY" -c "import $m;print(getattr($m,'__version__','?'))" 2>/dev/null) \
        && echo "  vorhanden  $m $v" || echo "  FEHLT      $m"
done

echo
echo "=== Installation der fehlenden Pakete ==="
rc=0
for m in scipy rasterio h5py sklearn matplotlib pytest; do
    "$PY" -c "import $m" >/dev/null 2>&1 && { echo "[skip] $m schon da"; continue; }
    pkg=$(pipname "$m")
    echo "[install] $m -> pip $pkg"
    "$PY" -m pip install --no-input --target "$DEPS" --upgrade "$pkg" 2>&1 | tail -3 \
        || { echo "  FEHLER bei $pkg - kein Netz auf diesem Knoten?"; rc=11; }
done

echo
echo "=== Ueberschattende Pakete entfernen ==="
# pip --target zieht auch Abhaengigkeiten mit, die das Environment laengst hat -
# etwa numpy. Da $DEPS im PYTHONPATH VOR site-packages steht, verdecken diese
# Kopien die Originale. Ein numpy-Sprung bricht kompilierte Erweiterungen wie
# torch erst zur Laufzeit, also im Trainingsjob und nicht hier. Deshalb bleibt
# nur, was dem Environment wirklich fehlt.
"$PY" - "$DEPS" <<'PRUNE'
import importlib.util, pathlib, shutil, sys
deps = pathlib.Path(sys.argv[1])
sys.path = [p for p in sys.path if not p.startswith(str(deps))]
names = {e.name[:-3] if e.name.endswith(".py") else e.name
         for e in deps.iterdir()
         if not e.name.endswith((".dist-info", ".egg-info", ".pth")) and e.name != "__pycache__"}
removed, kept = [], []
for n in sorted(names):
    try:
        spec = importlib.util.find_spec(n)
    except Exception:
        spec = None
    (removed if spec and spec.origin and not spec.origin.startswith(str(deps)) else kept).append(n)
for n in removed:
    for t in list(deps.glob(n + "*")):
        shutil.rmtree(t, ignore_errors=True) if t.is_dir() else t.unlink(missing_ok=True)
print("[prune] entfernt (Environment hat es selbst):", ", ".join(removed) or "nichts")
print("[prune] behalten (wirklich neu):", ", ".join(kept) or "nichts")
PRUNE

echo
echo "=== Abschlusspruefung ==="
"$PY" - <<'CHECK'
import importlib, sys
ok = True
for m in ("numpy", "torch", "scipy", "rasterio", "h5py", "sklearn", "matplotlib", "pytest"):
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FEHLT {m:12s} {type(e).__name__}: {e}")
        ok = False
try:
    import wildfire.eval.metrics  # noqa: F401
    print("  OK    wildfire importierbar")
except Exception as e:
    print(f"  FEHLT wildfire: {type(e).__name__}: {e}")
    ok = False
sys.exit(0 if ok else 12)
CHECK
rc2=$?

echo
echo "Inhalt von $DEPS:"; ls "$DEPS" | head -15
echo
if [ $rc -eq 0 ] && [ $rc2 -eq 0 ]; then
    echo "FERTIG - jetzt der Rauchtest:"
    echo "  SMOKE=1 sbatch wildfire/slurm/28_train_b2.sbatch"
else
    echo "UNVOLLSTAENDIG (pip rc=$rc, Pruefung rc=$rc2) - siehe oben."
fi
exit $(( rc || rc2 ))
