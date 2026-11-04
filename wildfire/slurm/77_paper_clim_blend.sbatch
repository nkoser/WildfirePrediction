#!/bin/bash
#SBATCH --job-name=wf_clim_blend
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Klimatologie gegen Modell im Paper-Protokoll: Scores der Ablationsmodelle LH, Historie 14 Tage
# und Contagion speichern (paper_metrics_ckpt --save-scores), dann clim_blend.
# Braucht 76 (clim.npz). Manifeste p17_paper_metrics_cache_G_next_clim, p25_clim_blend.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube
models=("sts_abl_lh:${CK}_abl_lh_s?_s?.pt" "sts_abl_hist14:${CK}_abl_hist14_s?_s?.pt" "sts_abl_cg:${CK}_abl_cg_s?_s?.pt")
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --models "${models[@]}" --tag _clim --save-scores || exit 1
"$PY" -m wildfire.paper.clim_blend --config "$CONFIG" --cache cache_G --scores scores_cache_G_next_clim.npz
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
