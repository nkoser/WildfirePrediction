#!/bin/bash
#SBATCH --job-name=wf_ablation
#SBATCH --array=0-95
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Ablation fuer das Paper (Tab. 3). Gemeinsames Protokoll: Wuerfel-Stichtage, Ziel D+1, d 64,
# 4 Bloecke, EMA 0.999, Batch 32, lr 3e-4, FESTE Epochenzahl (10 x 3000 Schritte) und die
# letzten EMA-Gewichte (--select last, keine Auswahl auf Val-Daten). Je Variante 3 Seeds;
# seed = i % 3, Variante v = i / 3, Tag _abl_<name>_s<seed>.
#   v0  lh             Referenz: Logistic Hazard
#   v1  bce            ohne Survival: ein Sigmoid je Horizont 2,7,8,16,30,31,100 (BCE), gemeinsamer Rumpf
#   v2-5 bce2/8/16/31  gestapelt: je Horizont ein eigenes Netz (nur dieser Horizont)
#   v6  lh_rate1       + Rate-Kopf P(Feuer an Tag tau), Gewicht 1       (braucht 73)
#   v7  lh_rate5       + Rate-Kopf, Gewicht 5                           (braucht 73)
#   v8  hist7          nur die letzten 7 Fenstertage
#   v9  hist14         nur die letzten 14 Fenstertage
#   v10 random         nur zufaellige Stichtage (echte Haeufigkeit)
#   v11 fire           nur an Feuer verankerte Stichtage
#   v12 paperonly      nur die 4.347 Paper-Stichproben (Cache-Stichtage), bis 30 Epochen, Auswahl Val-NLL
#   v13 dctm           Kopf Bernstein-DCTM
#   v14 cox            Kopf Cox (DeepSurv)
#   v15 deephit        Kopf DeepHit
#   v16-18 frac10/25/50  10 / 25 / 50 % der Wuerfel-Stichtage, Auswahl Val-NLL (Fruehstopp)
#   v19 joint          Joint-Attention ueber alle Token
#   v20 stem3d         3 x 3 x 3-Stem
#   Contagion v2 (K 7, 30 Beobachtungstage, Brennkurve), Rumpf aus v0, 3 Epochen (braucht v0):
#   v21 cg             Referenz
#   v22 cg_k1          K 1: keine Ausbreitung, nur Persistenz und Zuendung
#   v23 cg_k3          K 3
#   v24 cg_obs7        7 Beobachtungstage
#   v25 cg_geo         geometrische Brenndauer
#   v26 cg_scratch     ohne Warmstart, 10 Epochen (braucht v0 nicht)
#   v27 cg_rate1       + Rate-Kopf, Gewicht 1 (braucht v0 und 73)
#   v28 frac100        100 % mit Auswahl Val-NLL (Bezug fuer v16-18)
#   v29 paperonly_last   wie v12, aber fest 30 Epochen und letzte Gewichte (v12 waehlt Epoche 0: Val-NLL bei echter
#                        Haeufigkeit verschlechtert sich, weil die Paper-Stichproben zur Haelfte Feuer sind)
#   v30 paperonly_prauc  wie v12, Auswahl ueber die Paper-Val-PR-AUC (wie beim U-Net)
#   v31 oracle           Wetter-Oracle: + beobachtetes Wetter der Folgetage (--future-days 30, braucht 78;
#                        im Paper-Protokoll die dt Tage bis D). Obere Schranke, kein Modell fuer den Einsatz
# Flaechentreu (80): CACHE=cache_G_ea CUBE_DIR=paper_pixel_ea SURV_DIR=surv_eval_test_ea sbatch ...
#   (rate: 80 Index 3, oracle: 80 Index 4; Warmstart der Contagion-Varianten aus v0 desselben Caches)
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
seed=$((i % 3)); v=$((i / 3)); CK=/data/Wildfire/checkpoints/paper
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
warm=(--init-from "$CK/sts_${CACHE:-cache_G}_next_cube_abl_lh_s{seed}_s{seed}.pt" --init-backbone-only --epochs 3)
src=(--train-source cube)
case $v in
  0)  args=(--head lh --tag "_abl_lh_s${seed}") ;;
  1)  args=(--head bce --tag "_abl_bce_s${seed}") ;;
  2)  args=(--head bce --bce-taus 2 --tag "_abl_bce2_s${seed}") ;;
  3)  args=(--head bce --bce-taus 8 --tag "_abl_bce8_s${seed}") ;;
  4)  args=(--head bce --bce-taus 16 --tag "_abl_bce16_s${seed}") ;;
  5)  args=(--head bce --bce-taus 31 --tag "_abl_bce31_s${seed}") ;;
  6)  args=(--head lh --rate-weight 1 --tag "_abl_lh_rate1_s${seed}") ;;
  7)  args=(--head lh --rate-weight 5 --tag "_abl_lh_rate5_s${seed}") ;;
  8)  args=(--head lh --hist-days 7 --tag "_abl_hist7_s${seed}") ;;
  9)  args=(--head lh --hist-days 14 --tag "_abl_hist14_s${seed}") ;;
  10) args=(--head lh --cube-kind random --tag "_abl_random_s${seed}") ;;
  11) args=(--head lh --cube-kind fire --tag "_abl_fire_s${seed}") ;;
  12) src=(--train-source cache); args=(--head lh --select survnll --epochs 30 --steps-per-epoch 0 --tag "_abl_paperonly_s${seed}") ;;
  13) args=(--head dctm --tag "_abl_dctm_s${seed}") ;;
  14) args=(--head cox --tag "_abl_cox_s${seed}") ;;
  15) args=(--head deephit --tag "_abl_deephit_s${seed}") ;;
  16) args=(--head lh --train-frac 0.1 --select survnll --tag "_abl_frac10_s${seed}") ;;
  17) args=(--head lh --train-frac 0.25 --select survnll --tag "_abl_frac25_s${seed}") ;;
  18) args=(--head lh --train-frac 0.5 --select survnll --tag "_abl_frac50_s${seed}") ;;
  19) args=(--head lh --attn joint --tag "_abl_joint_s${seed}") ;;
  20) args=(--head lh --stem 3d --tag "_abl_stem3d_s${seed}") ;;
  21) args=("${cg[@]}" "${warm[@]}" --tag "_abl_cg_s${seed}") ;;
  22) args=(--head contagion --cg-kernel 1 --cg-obs-days 30 --cg-burn curve "${warm[@]}" --tag "_abl_cg_k1_s${seed}") ;;
  23) args=(--head contagion --cg-kernel 3 --cg-obs-days 30 --cg-burn curve "${warm[@]}" --tag "_abl_cg_k3_s${seed}") ;;
  24) args=(--head contagion --cg-kernel 7 --cg-obs-days 7 --cg-burn curve "${warm[@]}" --tag "_abl_cg_obs7_s${seed}") ;;
  25) args=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn geometric "${warm[@]}" --tag "_abl_cg_geo_s${seed}") ;;
  26) args=("${cg[@]}" --tag "_abl_cg_scratch_s${seed}") ;;
  27) args=("${cg[@]}" "${warm[@]}" --rate-weight 1 --tag "_abl_cg_rate1_s${seed}") ;;
  28) args=(--head lh --select survnll --tag "_abl_frac100_s${seed}") ;;
  29) src=(--train-source cache); args=(--head lh --select last --epochs 30 --steps-per-epoch 0 --tag "_abl_paperonly_last_s${seed}") ;;
  30) src=(--train-source cache); args=(--head lh --select prauc --epochs 30 --steps-per-epoch 0 --tag "_abl_paperonly_prauc_s${seed}") ;;
  31) args=(--head lh --future-days 30 --tag "_abl_oracle_s${seed}") ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_dctm --config "$CONFIG" --cache "${CACHE:-cache_G}" "${src[@]}" \
    --cube-dir "${CUBE_DIR:-paper_pixel}" --surv-dir "${SURV_DIR:-surv_eval_test}" \
    --target next_patch --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999 \
    --epochs 10 --steps-per-epoch 3000 --select last --seeds $seed --save-ckpt \
    "${args[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
