#!/bin/bash
#SBATCH --job-name=wf_sx_focus
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Modelle staerker auf Wetter, Brennstoff und Lage lenken (wildfire/paper/sts_focus.py), Protokoll wie 166:
#   0-2   lh_fdrop       LH, Feuer-Dropout p 0.5 (nur Training)
#   3-5   cg_fdrop       Contagion ohne Warmstart, Feuer-Dropout (auch der Ausbreitungspfad)
#   6-8   lh_resid       LH, Rest ueber der logistischen Feuerhistorie-Basis (Offset auf die Tages-Logits)
#   9-11  lh_ignw        LH, Zuendungsgewicht 4 fuer Stichproben ohne Feuer im Fenster
#   12-14 cg_ignw        Contagion, Zuendungsgewicht 4
#   15-17 lh_fdrop_mono  LH, Feuer-Dropout + Monotonie (Gewicht 1 wie lh_mono)
#   18-20 lh_fdropall    LH, Dropout von Feuermaske + LST + BT-Baendern gemeinsam
#   21-23 cg_fdropall    Contagion, dasselbe
#   24-47 dieselben acht Varianten mit Reflect-Padding im Stem (--pad reflect), Namen mit Suffix _rf, in der
#         Reihenfolge lh_fdrop_rf cg_fdrop_rf lh_resid_rf lh_ignw_rf cg_ignw_rf lh_fdrop_mono_rf lh_fdropall_rf cg_fdropall_rf
# Ausgaben NEU: checkpoints/paper/sts_extreme_<name>_s<seed>_s<seed>.pt, calib_cache_ext/, p35_sts_extreme_<name>_s<seed>,
# p44_sts_focus_<name>_s<seed>, derived_data/sts_focus/coef_<name>_s<seed>.npz (resid).
# Rauchtest: SFX=_smoke EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500"
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); rf=()
if [ "$v" -ge 8 ]; then v=$((v - 8)); rf=(--pad reflect); fi
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
case $v in
  0) n=lh_fdrop;      args=(--head lh --variants fdrop) ;;
  1) n=cg_fdrop;      args=("${cg[@]}" --variants fdrop) ;;
  2) n=lh_resid;      args=(--head lh --variants resid) ;;
  3) n=lh_ignw;       args=(--head lh --variants ignw) ;;
  4) n=cg_ignw;       args=("${cg[@]}" --variants ignw) ;;
  5) n=lh_fdrop_mono; args=(--head lh --variants fdrop --mono 1) ;;
  6) n=lh_fdropall;   args=(--head lh --variants fdropall) ;;
  7) n=cg_fdropall;   args=("${cg[@]}" --variants fdropall) ;;
esac
[ ${#rf[@]} -gt 0 ] && { n=${n}_rf; args+=("${rf[@]}"); }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_focus --config "$CONFIG" \
    --seeds $seed --tag "_${n}${SFX:-}_s${seed}" "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
