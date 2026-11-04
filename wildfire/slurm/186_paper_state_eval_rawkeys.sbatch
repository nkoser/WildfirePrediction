#!/bin/bash
#SBATCH --job-name=wf_state_raw
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Robustheit der Zustands-Auswertung gegen die Label-Schluessel: state_eval (wildfire/paper/state_eval.py) einmal
# mit allen Zeilen, Strata und Typen je mit eval_index (Fire-Index, valid-Maske) UND eval_raw (rohe MCD14A1, wie
# die Zielgroessen) auf denselben kalibrierten Vorhersagen und denselben Bootstrap-Ziehungen.
#   Checkpoints: lh, cg (scratch), utae, convlstm, lhbce10, cgbce10_scratch, cgw (warm), cgw_bce10, utae_bce10,
#                convlstm_bce10, ms_lam0, ms_lam1
#   gespeicherte kalibrierte Hazards aus msv2_eval (metrics/paper_multistate/msv2_surv.npz, Merkmale noetig):
#                msv2, msv2t, msf, msb, lhf, lhb, lhb10_scal, lhf_scal (Kalibrierung je Zustand), dazu lh und lhb10
#                als Abgleich (lh_npz, lhb10_npz muessen lh bzw. lhbce10 wiedergeben)
#   sbatch 186_...sbatch smoke  3.000 / 2.000 Stichtage, 20 Ziehungen -> state_eval_rawkeys_smoke_{index,raw}.json
#   sbatch 186_...sbatch        voll                                  -> state_eval_rawkeys_{index,raw}.json
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CK=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl
CKC=/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause
X=(--models lh cg utae convlstm
   --extra "lhbce10=${CK}_lhbce10_s?_s?.pt" "cgbce10=${CK}_cgbce10_scratch_s?_s?.pt" "cgw=${CK}_cg_s?_s?.pt"
           "cgw_bce10=${CK}_cgbce10_s?_s?.pt" "utae_bce10=${CK}_utae_bce10_s?_s?.pt"
           "convlstm_bce10=${CK}_convlstm_bce10_s?_s?.pt"
           "ms_lam0=${CKC}_ms_lam0_scratch_s?.pt" "ms_lam1=${CKC}_ms_lam1_scratch_s?.pt"
   --hazards-npz /data/Wildfire/metrics/paper_multistate/msv2_surv.npz
   --hazards msv2 msv2t msf msb lhf lhb lhb10_scal lhf_scal lh=lh_npz lhb10=lhb10_npz
   --keys index raw)
if [ "${1:-}" = smoke ]; then
  a=(--max-test 3000 --max-val 2000 --boot 20 --tag _rawkeys_smoke)
else
  a=(--tag _rawkeys)
fi
"$PY" -m wildfire.paper.state_eval --config "$CONFIG" "${X[@]}" "${a[@]}"
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
