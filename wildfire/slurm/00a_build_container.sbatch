#!/bin/bash
#SBATCH --job-name=wf_build_sif
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Baut wildfire.sif unprivilegiert (--fakeroot) aus wildfire/container/wildfire.def.
# Laeuft als einziger Job NICHT im Container - er baut ihn erst.
#
#   sbatch wildfire/slurm/00a_build_container.sbatch
#   OUT=/pfad/x.sif sbatch ...        anderes Ziel
#
# mksquashfs-Speicher hart begrenzt, sonst OOM -> leeres SIF (Muster aus
# BVM2027/sbatch/build_container.sbatch). Danach Pakettest im fertigen Image.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO=/home/nik/workspace/WildfirePrediction
DEF=$REPO/wildfire/container/wildfire.def
OUT=${OUT:-/scratch/shared/${USER}/containers/wildfire.sif}
WORK=/scratch/shared/${USER}/apptainer
mkdir -p "$(dirname "$OUT")" "$WORK/tmp" "$WORK/cache"
export APPTAINER_TMPDIR=$WORK/tmp APPTAINER_CACHEDIR=$WORK/cache
echo "[$(date -u +%FT%TZ)] build $DEF -> $OUT"
apptainer build --fakeroot --force \
    --mksquashfs-args="-mem 4G -processors ${SLURM_CPUS_ON_NODE:-8}" "$OUT" "$DEF"
[ "$(stat -c %s "$OUT")" -gt 1000000000 ] || { echo "FEHLER: SIF zu klein ($(stat -c %s "$OUT") B)"; exit 1; }
ls -lh "$OUT"
apptainer exec "$OUT" python -c "import torch, rasterio, pyproj, h5py, sklearn, yaml; print('ok torch', torch.__version__, 'rasterio', rasterio.__version__)"
echo "[$(date -u +%FT%TZ)] fertig."
