#!/bin/bash
#SBATCH --job-name=wf_hzbce_ev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Auswertung von 182 neben STS-LH (74 v0), alles im selben Aufruf:
#   surv_eval --calibrate --subsets           -> p16_surv_eval_hzbce.json
#   paper_metrics_ckpt (cache_G, Ziel D+1)     -> p17_paper_metrics_cache_G_next_hzbce.json
#   horizon_landmarks (PR-AUC tau 1..31 gegen den Stapel bce2/8/16/31, Monotonie, Testjahre)
#                                              -> metrics/paper_horizon/landmarks_hzbce.{json,npz}
# Die Checkpoints sind gewoehnliche LH-Checkpoints, kein Wrapper noetig.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
CK=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube
for f in /data/Wildfire/data_manifests/p16_surv_eval_hzbce.json /data/Wildfire/data_manifests/p17_paper_metrics_cache_G_next_hzbce.json; do
  if [ -e "$f" ]; then echo "ABBRUCH: $f existiert schon"; exit 1; fi
done
m=("sts_abl_lhbce03:${CK}_abl_lhbce03_s?_s?.pt" "sts_abl_lhbce10:${CK}_abl_lhbce10_s?_s?.pt"
   "sts_abl_lh:${CK}_abl_lh_s?_s?.pt")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --calibrate --subsets \
    --models "${m[@]}" --tag _hzbce || exit 1
"$PY" -m wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
    --models "${m[@]}" --tag _hzbce || exit 1
"$PY" -m wildfire.paper.horizon_landmarks --config "$CONFIG" --skip joint classic --tag _hzbce \
    --curves "lh=${CK}_abl_lh_s?_s?.pt" "lhbce03=${CK}_abl_lhbce03_s?_s?.pt" "lhbce10=${CK}_abl_lhbce10_s?_s?.pt"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
