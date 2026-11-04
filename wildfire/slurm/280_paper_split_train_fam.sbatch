#!/bin/bash
#SBATCH --job-name=wf_split_fam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Modellfamilien des zeitlichen Splits auf den Kandidaten-Splits, je Familie FAM ein Array-Job. Module und
# Einstellungen exakt wie in den genannten sbatch-Skripten, nur Daten und Cache des Splits:
#   --cache cache_G_<yearstrat|blockstrat> --cube-dir paper_pixel_<sp> --surv-dir surv_eval_<sp> (+ Familien-Daten).
# Index i: Split sp = i / N (0 ys, 1 bs), lokal j = i % N, Variante v = j / 3, Seed = j % 3 (ausser msv2).
#   FAM=ms      N 12  wie 170 (multistate train) und 91 v0/v1 (contagion_cause):        braucht 276 k6 (Ursachen)
#                     v0 ms_lam0_scratch  v1 ms_lam1_scratch  v2 lam0_scratch  v3 lam1_scratch
#   FAM=msv2    N 16  wie 201: j 0-11 msv2/msv2t/lhf/msf je 3 Seeds, 12 lhb Seed 0, 13-15 msb:  braucht 276 k7
#   FAM=spread  N 6   wie 256: cg_tfonly, cg_tfonlybig                                    braucht 276 k3 (Ring)
#   FAM=gpt     N 9   wie 253: gpt_tf, gpt_ss05, gpt_free                                  braucht 276 k4 (future31)
#   FAM=hzbce   N 6   wie 182: lhbce03, lhbce10
#   FAM=bce10   N 12  wie 195: cgbce10_scratch, cgbce10 (warm aus lhbce10 dieses Splits), utae_bce10, convlstm_bce10
#   FAM=vision  N 6   wie 130: utae, convlstm
#   FAM=episode N 3   wie 215: cg_episode
# Checkpoint-Namen tragen den Split: ueber den Cache (paper_cause, paper_msv2, sts_<cache>_next_cube_abl_*) oder
# ueber den Tag _<sp>_<name>_s<seed> (sts_spread_tfonly, fire_gpt, sts_extreme). Jeder Lauf bricht ab, wenn sein
# Checkpoint schon existiert.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
declare -A NN=([ms]=12 [msv2]=16 [spread]=6 [gpt]=9 [hzbce]=6 [bce10]=12 [vision]=6 [episode]=3)
N=${NN[${FAM:?FAM setzen}]:?unbekannte FAM}
s=$((i / N)); j=$((i % N)); v=$((j / 3)); seed=$((j % 3))
sps=(ys bs); caches=(cache_G_yearstrat cache_G_blockstrat); sp=${sps[$s]}; cache=${caches[$s]}
CK=/data/Wildfire/checkpoints/paper
data=(--cache "$cache" --cube-dir "paper_pixel_${sp}" --surv-dir "surv_eval_${sp}")
cg=(--head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve)
dctm=(--train-source cube --target next_patch --d 64 --depth 4 --bs 32 --lr 3e-4 --warmup 500 --ema 0.999
      --select last --seeds $seed --save-ckpt)
chk() { if ls $1 >/dev/null 2>&1; then echo "ABBRUCH: $1 existiert schon"; exit 1; fi; }
case $FAM in
  ms)
    tags=(_ms_lam0_scratch _ms_lam1_scratch _lam0_scratch _lam1_scratch); lams=(0 1 0 1)
    W=(-m wildfire.paper.multistate train); [ $v -ge 2 ] && W=(-m wildfire.paper.contagion_cause)
    chk "/data/Wildfire/checkpoints/paper_cause/sts_${cache}_next_cube_cause${tags[$v]}_s${seed}.pt"
    cmd=("$PY" "${W[@]}" --config "$CONFIG" --cache "$cache" --cube-dir "paper_pixel_${sp}"
         --labels "train_cause_${sp}.npz" --seed $seed --init-from= --epochs 10 --lam "${lams[$v]}" --tag "${tags[$v]}") ;;
  msv2)
    if [ $j -eq 12 ]; then k=lhb; seed=0; elif [ $j -ge 13 ]; then k=msb; seed=$((j - 13))
    else ks=(msv2 msv2t lhf msf); k=${ks[$v]}; fi
    case $k in msv2) a=(--kind msv2) ;; msv2t) a=(--kind msv2 --lam-type 0.3) ;; lhf) a=(--kind lhf) ;;
               msf) a=(--kind msf) ;; lhb) a=(--kind lh) ;; msb) a=(--kind ms) ;; esac
    chk "/data/Wildfire/checkpoints/paper_msv2/sts_${cache}_next_cube_msv2_${k}_s${seed}.pt"
    cmd=("$PY" -m wildfire.paper.multistate_v2 train --config "$CONFIG" --beta 1 --cache "$cache"
         --cube-dir "paper_pixel_${sp}" --feat-tag "_${sp}" "${a[@]}" --seed $seed --tag "_msv2_$k") ;;
  spread)
    ns=(cg_tfonly cg_tfonlybig); n=${ns[$v]}
    as=("--spread ell --cg-kernel 11 --tf-weight 1.0 --mf-weight 0 --big-share 0"
        "--spread ell --cg-kernel 11 --tf-weight 1.0 --mf-weight 0 --big-thr 5 --big-days 3 --big-share 0.5")
    tag="_${sp}_${n}_s${seed}"; chk "$CK/sts_spread_tfonly${tag}_s*.pt"
    cmd=("$PY" -m wildfire.paper.sts_spread_tfonly --config "$CONFIG" "${data[@]}" --ring-dir "paper_pixel_${sp}_ring5"
         --seeds $seed --tag "$tag" ${as[$v]}) ;;
  gpt)
    ns=(gpt_tf gpt_ss05 gpt_free); as=("--mode tf" "--mode ss --ss-prob 0.5" "--mode free"); n=${ns[$v]}
    tag="_${sp}_${n}_s${seed}"; chk "$CK/fire_gpt${tag}_s*.pt"
    cmd=("$PY" -m wildfire.paper.fire_gpt --config "$CONFIG" --cache "$cache" --cube-dir "paper_pixel_${sp}"
         --fut-dir "paper_pixel_${sp}_future31" --seeds $seed --tag "$tag" ${as[$v]}) ;;
  hzbce)
    betas=(0.3 1.0); ns=(lhbce03 lhbce10); tag="_abl_${ns[$v]}_s${seed}"
    chk "$CK/sts_${cache}_next_cube${tag}_s${seed}.pt"
    cmd=("$PY" -m wildfire.paper.horizon_bce train --hz-beta "${betas[$v]}" --hz-taus 1 7 14 30 --config "$CONFIG"
         "${data[@]}" "${dctm[@]}" --head lh --epochs 10 --steps-per-epoch 3000 --tag "$tag") ;;
  bce10)
    ns=(cgbce10_scratch cgbce10 utae_bce10 convlstm_bce10); n=${ns[$v]}; tag="_abl_${n}_s${seed}"
    warm=(--init-from "$CK/sts_${cache}_next_cube_abl_lhbce10_s{seed}_s{seed}.pt" --init-backbone-only --epochs 3)
    case $v in 0) a=("${cg[@]}") ;; 1) a=("${cg[@]}" "${warm[@]}") ;;
               2) a=(--backbone utae --head lh) ;; 3) a=(--backbone convlstm --head lh) ;; esac
    chk "$CK/sts_${cache}_next_cube${tag}_s${seed}.pt"
    cmd=("$PY" -m wildfire.paper.horizon_bce_any train --hz-beta 1.0 --hz-taus 1 7 14 30 --config "$CONFIG"
         "${data[@]}" "${dctm[@]}" --epochs 10 --steps-per-epoch 3000 "${a[@]}" --tag "$tag") ;;
  vision)
    bbs=(utae convlstm); bb=${bbs[$v]}; tag="_abl_${bb}_s${seed}"
    chk "$CK/sts_${cache}_next_cube${tag}_s${seed}.pt"
    cmd=("$PY" -m wildfire.paper.vision_backbones train "$bb" --config "$CONFIG" "${data[@]}" "${dctm[@]}"
         --head lh --epochs 10 --steps-per-epoch 3000 --tag "$tag") ;;
  episode)
    tag="_${sp}_cg_episode_s${seed}"; chk "$CK/sts_extreme${tag}_s*.pt"
    cmd=("$PY" -m wildfire.paper.sts_episode --config "$CONFIG" "${data[@]}" --ep-gap "${EP_GAP:-2}" "${cg[@]}"
         --seeds $seed --tag "$tag") ;;
esac
echo "[split_fam] $FAM $sp j=$j: ${cmd[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${cmd[@]}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
