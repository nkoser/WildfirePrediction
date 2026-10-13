#!/bin/bash
#SBATCH --job-name=wf_fetch_artifacts
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Externe Daten, die NICHT im MPCFire-Datensatz stehen, nach $EXP/artifacts:
#
#   NFDB_point_txt.zip                      Canadian National Fire Database,
#                                           Punktmeldungen (Labels, Schritt 30)
#   ne_50m_admin_0_countries.geojson        Landesgrenze Kanada (Schritt 10)
#   ne_50m_admin_1_states_provinces.geojson Provinzen (Schritt 13)
#
# Auf dem alten Cluster wurden diese von Hand geladen. NFDB wird jaehrlich
# aktualisiert; eine neue Version kann andere Zaehlungen ergeben als im
# Artifact. Deshalb Download-Zeitpunkt und sha256 nach artifacts/SHA256SUMS.
# Vorhandene Dateien werden NICHT ueberschrieben (FORCE=1 erzwingt es).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
A="$EXP/artifacts"; mkdir -p "$A"
NFDB_URL="${NFDB_URL:-https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_txt.zip}"
NE=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson
rc=0
fetch() {   # url ziel
    if [ -s "$2" ] && [ -z "${FORCE:-}" ]; then echo "[skip] $(basename "$2") vorhanden"; return; fi
    "$PY" - "$1" "$2" <<'PYDL' || rc=1
import sys, urllib.request, shutil
url, dst = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url, timeout=300) as r, open(dst + ".part", "wb") as f:
    shutil.copyfileobj(r, f)
import os; os.replace(dst + ".part", dst)
print(f"[ok] {url} -> {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
PYDL
}
fetch "$NFDB_URL" "$A/NFDB_point_txt.zip"
fetch "$NE/ne_50m_admin_0_countries.geojson" "$A/ne_50m_admin_0_countries.geojson"
fetch "$NE/ne_50m_admin_1_states_provinces.geojson" "$A/ne_50m_admin_1_states_provinces.geojson"
{ echo "# $(date -u +%FT%TZ) job=${SLURM_JOB_ID:-?} nfdb_url=$NFDB_URL"
  cd "$A" && sha256sum NFDB_point_txt.zip ne_50m_*.geojson; } >> "$A/SHA256SUMS"
tail -4 "$A/SHA256SUMS"
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
