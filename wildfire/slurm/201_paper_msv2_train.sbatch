#!/bin/bash
#SBATCH --job-name=wf_msv2
#SBATCH --array=0-12%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Multi-State v2 und Kontrollen (wildfire/paper/multistate_v2.py), alle von Grund auf, Horizont-BCE beta 1,
# 10 x 3.000 Schritte, Batch 32, letzte EMA-Gewichte -> checkpoints/paper_msv2/sts_cache_G_next_cube_msv2_<v>_s<seed>.pt
#   0- 2  msv2     Merkmale + Rueckkehr ueber die Pause            (Bausteine 1, 2, 4)
#   3- 5  msv2t    msv2 + Ursachen-Supervision lam_type 0.3         (Baustein 3)
#   6- 8  lhf      STS-LH + Merkmale                                (Kontrolle: Merkmale ohne Multi-State-Struktur)
#   9-11  msf      Multi-State v1-Rueckkehr + Merkmale              (Baustein 1 allein)
#  12     lhb      STS-LH in dieser Schleife, Seed 0                (Abgleich mit _abl_lhbce10 aus sts_dctm)
#  13-15  msb      Multi-State v1-Struktur + beta 1 (ohne Merkmale)  (nur mit --array=13-15 starten)
#   SMOKE=1 sbatch --array=0 201_...sbatch    20 Schritte, 2.000 Stichtage, Merkmale _smoke, Tag _msv2_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
W=(-m wildfire.paper.multistate_v2 train --config "$CONFIG" --beta 1)
if [ -n "${SMOKE:-}" ]; then
  rc=0
  for k in msv2 lhf; do
    "$PY" "${W[@]}" --kind $k --seed 0 --epochs 1 --steps-per-epoch 20 --limit 2000 --feat-tag _smoke \
        --lam-type $([ $k = msv2 ] && echo 0.3 || echo 0) --tag _msv2_smoke_$k || rc=$?
  done
  echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
fi
if [ "$i" -eq 12 ]; then v=lhb; seed=0
elif [ "$i" -ge 13 ]; then v=msb; seed=$((i - 13))
else vs=(msv2 msv2t lhf msf); v=${vs[$((i / 3))]}; seed=$((i % 3)); fi
case $v in
  msv2)  a=(--kind msv2) ;;
  msv2t) a=(--kind msv2 --lam-type 0.3) ;;
  lhf)   a=(--kind lhf) ;;
  msf)   a=(--kind msf) ;;
  lhb)   a=(--kind lh) ;;
  msb)   a=(--kind ms) ;;
esac
"$PY" "${W[@]}" "${a[@]}" --seed $seed --tag "_msv2_$v"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
