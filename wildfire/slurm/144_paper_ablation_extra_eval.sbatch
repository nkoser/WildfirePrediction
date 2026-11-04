#!/bin/bash
#SBATCH --job-name=wf_abl_extra_ev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Auswertung der Zusatz-Ablation (143) wie 75: surv_eval (100.000 zufaellige Stichtage, --subsets) und
# paper_metrics_ckpt (Paper-Protokoll) -> Manifeste p16_surv_eval_abl_g7, p17_paper_metrics_cache_G_next_abl_g7
# (Gruppe 7 ist neu; 75 hat die Gruppen 0-6).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube
for f in /data/Wildfire/data_manifests/p16_surv_eval_abl_g7.json /data/Wildfire/data_manifests/p17_paper_metrics_cache_G_next_abl_g7.json; do
  if [ -e "$f" ]; then echo "ABBRUCH: $f existiert schon"; exit 1; fi
done
models=()
for n in local pw0 d128; do models+=("sts_abl_${n}:${CK}_abl_${n}_s?_s?.pt"); done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --subsets \
    --models "${models[@]}" --tag _abl_g7 || exit 1
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --models "${models[@]}" --tag _abl_g7
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
