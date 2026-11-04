#!/bin/bash
#SBATCH --job-name=wf_bce10_ev
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Auswertung von 195 (Horizont-BCE beta 1.0) neben den alten Zeilen ohne Zusatzterm, gleiche Stichtage und
# Kalibrierung je Aufruf. Laden ueber den Wrapper paper/vision_backbones.py (erkennt U-TAE / ConvLSTM, sonst wie bisher).
#   0  surv_eval --calibrate --subsets -> p16_surv_eval_bce10rows.json
#      paper_metrics_ckpt (cache_G, Ziel D+1) -> p17_paper_metrics_cache_G_next_bce10rows.json
#   1  horizon_landmarks --curves (PR-AUC tau 1..31 gegen den Stapel) -> metrics/paper_horizon/landmarks_bce10rows.*
#   2  state_eval (Typ-AUC@30, NLL je Zustand) -> metrics/paper_multistate/state_eval_bce10rows.json
#   sbatch --array=0-2 196_...sbatch smoke   kleine Teilmengen, Tag _bce10rows_smoke (nur 1 und 2; 0 hat kein Limit)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
j=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
CK=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube
W=(-m wildfire.paper.vision_backbones)
names=(lh lhbce10 cg_scratch cgbce10_scratch cg cgbce10 utae utae_bce10 convlstm convlstm_bce10)
tag=_bce10rows; lim=(); sm=0
if [ "${1:-}" = smoke ]; then tag=_bce10rows_smoke; lim=(--max-test 3000 --max-val 2000 --boot 20); sm=1; fi
pat() {   # Checkpoint-Muster; im Rauchtest die Rauchtest-Checkpoints aus 195 fuer die neuen Zeilen
  case $1 in *bce10*) [ $sm = 1 ] && [ "$1" != lhbce10 ] && { echo "${CK}_abl_$1_smoke_s?.pt"; return; } ;; esac
  echo "${CK}_abl_$1_s?_s?.pt"
}
m=(); c=(); for n in "${names[@]}"; do m+=("sts_abl_${n}:$(pat "$n")"); c+=("${n}=$(pat "$n")"); done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $j in
  0)
    if [ "${1:-}" = smoke ]; then echo "Teil 0 ohne Rauchtest"; exit 0; fi
    "$PY" "${W[@]}" wildfire.paper.surv_eval --config "$CONFIG" --out-name surv_eval_test --calibrate --subsets \
        --models "${m[@]}" --tag "$tag" || exit 1
    "$PY" "${W[@]}" wildfire.paper.paper_metrics_ckpt --config "$CONFIG" --cache cache_G --rec-run _next_paperhp_w0 \
        --models "${m[@]}" --tag "$tag" ;;
  1)
    "$PY" "${W[@]}" wildfire.paper.horizon_landmarks --config "$CONFIG" --skip joint classic lincg --tag "$tag" \
        "${lim[@]}" --curves "${c[@]}" ;;
  2)
    # state_eval kennt schon lh, cg (= cg_scratch), utae, convlstm; warm gestartetes cg heisst hier cgw
    ex=(); for n in "${names[@]}"; do case $n in
      lh|utae|convlstm|cg_scratch) ;;
      cg) ex+=("cgw=$(pat cg)") ;;
      cgbce10) ex+=("cgw_bce10=$(pat cgbce10)") ;;
      *) ex+=("${n}=$(pat "$n")") ;;
    esac; done
    "$PY" -m wildfire.paper.state_eval --config "$CONFIG" --models lh cg utae convlstm --extra "${ex[@]}" \
        --tag "$tag" "${lim[@]}" ;;
esac
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
