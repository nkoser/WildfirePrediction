# Wird von allen Audit-sbatch-Skripten gesourced.
# Haelt Umgebung, Pfade und Logging an einer Stelle.
#
# ACHTUNG, HOST- GEGEN CONTAINER-PFAD:
# --output und --error werden von Slurm auf dem HOST aufgeloest, bevor der
# Container existiert. Dort muss also der Host-Pfad stehen:
#     /raid/data/nik/Wildfire/logs/%x_%j.out
# NICHT /data/Wildfire/logs/... - dieses Verzeichnis gibt es nur IM Container.
# Steht dort der Container-Pfad, kann slurmd die Ausgabedatei nicht anlegen,
# der Job stirbt sofort beim Start, verschwindet aus der Queue und
# hinterlaesst kein Log - weil das Log genau das ist, was fehlschlaegt.
# Alles Uebrige in diesen Skripten (EXP, CONFIG, Datenpfade) laeuft dagegen
# IM Container und benutzt korrekt /data.
set -uo pipefail

REPO="/workspace/WildfirePrediction"
EXP="/data/Wildfire"
CONFIG="${CONFIG:-$REPO/wildfire/configs/audit.yaml}"

# Interpreterwahl. Das Projekt laeuft im Image /raid/data/nik/vesselgpt_5:
#
#   /miniconda3/bin/conda                Env "vmtk_2"
#   /miniconda3/envs/vmtk_2/bin/python   Python 3.11.8, numpy, scipy, h5py,
#                                        sklearn, matplotlib, pytest, torch
#                                        2.10+cu128  -- nur rasterio fehlt und
#                                        kommt aus $DEPS
#
# Das ist exakt die Umgebung, in der auch interaktiv gerechnet wurde. Der
# fruehere Versuch mit /raid/data/nik/SR scheiterte, weil dessen Env "sr_env"
# (Python 3.12) kein h5py hat und die kompilierten Pakete in $DEPS fuer
# cpython-311 gebaut sind.
#
# Zwei Fehler der ersten Fassung, beide von Job 5845 aufgedeckt:
#
#  1. Gesucht wurde nur nach ".../bin/python". Im Image heisst die Datei
#     "python3.12". Es wurde also KEIN Interpreter gefunden.
#  2. Verlangt wurden numpy UND h5py UND scipy. h5py braucht aber nur das
#     Audit (s4/s5/s6), nicht das Training. Ein fehlendes h5py hat damit
#     Jobs abgebrochen, die es nie benutzt haetten.
#
# Deshalb jetzt: harte Mindestanforderung ist allein numpy. Was sonst fehlt,
# wird gemeldet - der Job entscheidet selbst, ob es ihn betrifft.
#
# Zusatzpakete liegen persistent in $DEPS und werden per PYTHONPATH
# vorangestellt (Vorgehen wie AneuG/setup_a1_deps.sbatch). WICHTIG: Kompilierte
# Pakete sind an die Python-Minor-Version gebunden - ein rasterio fuer cp311
# laesst sich unter 3.12 nicht importieren. $DEPS traegt die Version deshalb im
# Namen, damit sich die Baeume nie vermischen.
#
#   PY=/pfad/zu/python sbatch <skript>    uebersteuert die Wahl
#   DEPS=/pfad          sbatch <skript>   uebersteuert das Paketverzeichnis

if [ -z "${PY:-}" ]; then
    for cand in \
            /miniconda3/envs/vmtk_2/bin/python \
            /opt/miniconda3/envs/sr_env/bin/python3.12 \
            /opt/miniconda3/envs/sr_env/bin/python \
            $(ls -d /opt/miniconda3/envs/*/bin/python3.* /miniconda3/envs/*/bin/python3.* \
                    /opt/miniconda3/envs/*/bin/python  /miniconda3/envs/*/bin/python 2>/dev/null) \
            /opt/miniconda3/bin/python3.13 /usr/bin/python3; do
        [ -x "$cand" ] || continue
        case "$cand" in *-config) continue ;; esac
        "$cand" -c "import numpy" >/dev/null 2>&1 && { PY="$cand"; break; }
    done
fi
if [ -z "${PY:-}" ]; then
    echo "ABBRUCH: kein Interpreter mit numpy gefunden." >&2
    echo "  Gepruefte Kandidaten unter /opt/miniconda3 und /miniconda3:" >&2
    ls -d /opt/miniconda3/envs/*/bin/python3.* /miniconda3/envs/*/bin/python3.* 2>/dev/null >&2
    echo "  Abhilfe: PY=<pfad> sbatch <skript>, oder 01c_setup_deps.sbatch laufen lassen." >&2
    exit 2
fi

PYTAG=$("$PY" -c "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
DEPS="${DEPS:-$EXP/pydeps-$PYTAG}"
export PYTHONPATH="$DEPS:$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$EXP/logs" "$EXP/data_manifests" "$EXP/reports" "$EXP/figures"

echo "[$(date -u +%FT%TZ)] host=$(hostname) job=${SLURM_JOB_ID:-interactive} py=$PY"
echo "[cfg] $CONFIG"
echo "[env] $PY  ($PYTAG)   DEPS=$DEPS"
# Nur berichten, nicht abbrechen: welcher Job welches Paket braucht, weiss er selbst.
"$PY" - <<'PYCHK'
import importlib.util
missing = [m for m in ("scipy", "rasterio", "h5py", "torch", "sklearn", "matplotlib", "pytest")
           if importlib.util.find_spec(m) is None]
print("[env] fehlende optionale Pakete: " + (", ".join(missing) if missing else "keine"))
PYCHK
