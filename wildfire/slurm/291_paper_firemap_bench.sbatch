#!/bin/bash
#SBATCH --job-name=wf_firemap_bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Feuerkarten-Benchmark ueber alle groesseren Feuer von val (2020-22) und test (2023-25), temporaler Split
# (wildfire/paper/firemap_bench.py) -> metrics/firemap_bench${TAG}/.
#   STAGE=events  (ohne Array)  Ereignisliste events_val/test.json
#   STAGE=run     Array: split = i % 2 (0 val, 1 test), Modell = MODELS[i / 2]
#   STAGE=report  (ohne Array)  summary.md/json
# MODELS: 0 dist, 1-7 Spread-Kerne, 8-10 Self-Forcing, 11-13 pixelweise ohne Kopplung (indep:), 14 cg_scratch Regionssimulation
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper
MODELS=(
  "dist"
  "cg_ell11=$CK/sts_spread_cg_ell11_s?_s?.pt"
  "cg_ell11tf10=$CK/sts_spread_tf_cg_ell11tf10_s?_s?.pt"
  "cg_bigtf30=$CK/sts_spread_big_cg_bigtf30_s?_s?.pt"
  "cg_tfonly=$CK/sts_spread_tfonly_cg_tfonly_s?_s?.pt"
  "cg_tfonlybig=$CK/sts_spread_tfonly_cg_tfonlybig_s?_s?.pt"
  "cg_tfmf01=$CK/sts_spread_tfonly_cg_tfmf01_s?_s?.pt"
  "cg_tfmf03=$CK/sts_spread_tfonly_cg_tfmf03_s?_s?.pt"
  "cg_sf05=$CK/sts_spread_sf_cg_sf05_s?_s?.pt"
  "cg_sf10=$CK/sts_spread_sf_cg_sf10_s?_s?.pt"
  "cg_sf10big=$CK/sts_spread_sf_cg_sf10big_s?_s?.pt"
  "lh_indep=indep:$CK/sts_cache_G_next_cube_abl_lh_s?_s?.pt"
  "cgscratch_indep=indep:$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"
  "cg_ell11_indep=indep:$CK/sts_spread_cg_ell11_s?_s?.pt"
  "cg_scratch=$CK/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"
  "cg_sfiou05=$CK/sts_spread_sfiou_cg_sfiou05_s?_s?.pt"
  "cg_sfiou20=$CK/sts_spread_sfiou_cg_sfiou20_s?_s?.pt"
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case ${STAGE:?STAGE=events|run|report} in
  run)
    i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
    sp=$([ $((i % 2)) -eq 0 ] && echo val || echo test)
    "$PY" -m wildfire.paper.firemap_bench run --config "$CONFIG" --tag "${TAG:-}" --split $sp --model "${MODELS[$((i / 2))]}" ${EXTRA:-} ;;
  *)
    "$PY" -m wildfire.paper.firemap_bench "$STAGE" --config "$CONFIG" --tag "${TAG:-}" ${EXTRA:-} ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
