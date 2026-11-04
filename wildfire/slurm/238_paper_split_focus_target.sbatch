#!/bin/bash
#SBATCH --job-name=wf_split_fet
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Varianten-Ablation mit Paper-Ziel D (wildfire/paper/focus_eval_target.py, nur Auswertung derselben Checkpoints):
#   0  zeitlich  cache_G,            paper_pixel,    Modelle ohne Praefix (159/166/148)   -> _D_t
#   1  Jahre     cache_G_yearstrat,  paper_pixel_ys, Modelle ys_* (234)                  -> _D_ys
#   2  Bloecke   cache_G_blockstrat, paper_pixel_bs, Modelle bs_* (234)                  -> _D_bs
#   3  zeitlich, Ziel D+1: Selbsttest, muss metrics/focus_eval{,_cgfdmono} wiederholen -> _D1_t
# Ausgaben NEU: metrics/focus_eval_target<tag>/, Manifest p52_focus_eval_target<tag>.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
pre=("" "ys_" "bs_" ""); caches=(cache_G cache_G_yearstrat cache_G_blockstrat cache_G)
cubes=(paper_pixel paper_pixel_ys paper_pixel_bs paper_pixel); tags=(_D_t _D_ys _D_bs _D1_t)
tg=cache; [ "$i" -eq 3 ] && tg=next_patch
p=${pre[$i]}
M="${p}lh_ref ${p}cg_ref ${p}cg_mono ${p}lh_fdrop:focus=fdrop ${p}lh_resid:focus=resid ${p}lh_ignw:focus=ignw"
M="$M ${p}lh_fdropall:focus=fdropall ${p}lh_fdrop_mono:focus=fdrop ${p}lh_fdrop_mono_rf:focus=fdrop:pad=reflect"
M="$M ${p}cg_fdrop:focus=fdrop ${p}cg_ignw:focus=ignw ${p}cg_fdrop_mono:focus=fdrop ${p}cg_fdrop_mono_rf:focus=fdrop:pad=reflect"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.focus_eval_target --config "$CONFIG" --target $tg --cache "${caches[$i]}" \
    --cube-dir "${cubes[$i]}" --tag "${tags[$i]}${SFX:-}" --models $M ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
