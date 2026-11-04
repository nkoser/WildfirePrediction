#!/bin/bash
#SBATCH --job-name=wf_pack_transfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Packt alles fuer den Umzug auf einen anderen Server (siehe MIGRATION_NEW_SERVER.md)
# nach /scratch/shared/nik/transfer:
#   wildfire_core.tar     Repo (inkl. .git und uncommitteter Aenderungen), wildfire.sif,
#                         Claude-Memory, $EXP ohne die grossen derived_data-Ordner
#   mpcfire_caches.tar    MPCFire cache_A/G/H/Y + cache_G_*-Splitvarianten + README
#   *.sha256, *.list      Pruefsummen und Inhaltslisten
# Grosse derived_data-Ordner (cube, paper*, surv_eval*, calib_cache_ext) sind NICHT
# drin - per rsync uebertragen oder neu bauen. Ohne Kompression: npz/sif sind schon
# komprimiert. Laufende Jobs schreiben weiter -> Stand = Zeitpunkt des Packens.
# Vorher auf dem Host:  mkdir -p /scratch/shared/nik/transfer
#   WF_EXTRA_BINDS=/scratch/shared/nik/transfer:/transfer,/scratch/shared/nik/containers:/containers:ro \
#     sbatch wildfire/slurm/zz_pack_transfer.sbatch
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
set -uo pipefail
OUT=/transfer
[ -d "$OUT" ] && [ -d /containers ] || { echo "ABBRUCH: WF_EXTRA_BINDS fuer /transfer und /containers fehlen" >&2; exit 2; }
BIG=(cube paper paper_pixel paper_pixel_bs paper_pixel_ys paper_pixel_ea calib_cache_ext
     surv_eval_test surv_eval_test_ea surv_eval_ys surv_eval_bs paper_extra paper_surv)
ex=(--exclude=Wildfire/tmp)
for d in "${BIG[@]}"; do ex+=(--exclude="Wildfire/derived_data/$d"); done

echo "[$(date -u +%FT%TZ)] core"
tar --warning=no-file-changed "${ex[@]}" -cf "$OUT/wildfire_core.tar" \
    -C /workspace WildfirePrediction \
    -C /data Wildfire \
    -C / containers/wildfire.sif \
    -C "$OUT" claude_memory
rc=$?; [ $rc -gt 1 ] && { echo "tar core exit $rc" >&2; exit $rc; }

echo "[$(date -u +%FT%TZ)] caches"
tar -cf "$OUT/mpcfire_caches.tar" -C /firedata MPCFire/README.md \
    $(cd /firedata && ls -d MPCFire/cache_*) || exit $?

cd "$OUT"
for t in wildfire_core.tar mpcfire_caches.tar; do
    tar -tvf "$t" > "${t%.tar}.list"
    sha256sum "$t" > "$t.sha256"
done
printf '%s\n' "${BIG[@]}" > derived_data_NOT_included.txt
ls -lh "$OUT"
echo "[$(date -u +%FT%TZ)] exit 0"
