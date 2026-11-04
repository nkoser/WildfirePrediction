#!/bin/bash
#SBATCH --job-name=wf_split_unet
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# 3D-U-Net-Nachbau mit den Kanalsaetzen der Eingabe-Ablation auf den Kandidaten-Splits, Paper-Hyperparameter und
# Run-Namen wie auf dem zeitlichen Split (85, 211, 213), nur --cache cache_G_<yearstrat|blockstrat>:
# Satz c = i % 5 (no_fire no_modis no_thermal fire_only weather), Ziel z = (i / 5) % 2 (0 D, 1 D+1), Split = i / 10.
#   D:   --end 2 8 16 31,                      Run _paperhp_<satz>
#   D+1: --target next_patch --end 1 7 15 30,  Run _next_paperhp_w0_<nofire|nomodis|no_thermal|fire_only|weather>
# no_fire / no_modis / weather mit reconstruct (channels.py), no_thermal / fire_only mit reconstruct_inputs.
# 3 Seeds je dt. Ausgaben NEU: p2_reconstruct_<cache>_cache<run>, predictions/paper/rec_<cache>_cache<run>.npz.
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
CONFIG=/workspace/WildfirePrediction/wildfire/configs/audit_splitsv2.yaml
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
i=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
c=$((i % 5)); z=$(((i / 5) % 2)); s=$((i / 10))
caches=(cache_G_yearstrat cache_G_blockstrat); cache=${caches[$s]}
sets=(no_fire no_modis no_thermal fire_only weather); cs=${sets[$c]}
d1=(nofire nomodis no_thermal fire_only weather)
mods=(reconstruct reconstruct reconstruct_inputs reconstruct_inputs reconstruct)
if (( z == 0 )); then
  tgt=(--end 2 8 16 31); rn="_paperhp_${cs}${SFX:-}"
else
  tgt=(--target next_patch --end 1 7 15 30); rn="_next_paperhp_w0_${d1[$c]}${SFX:-}"
fi
if [[ -e "$EXP/data_manifests/p2_reconstruct_${cache}_cache${rn}.json" ]]; then
  echo "ABBRUCH: Manifest p2_reconstruct_${cache}_cache${rn}.json existiert schon"; exit 1
fi
echo "[split_unet] $cache $cs Ziel $z -> Run $rn (${mods[$c]})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.${mods[$c]} --config "$CONFIG" --cache "$cache" --norm cache \
    --seeds 0 1 2 --channels "$cs" --T 30 "${tgt[@]}" --bs 512 --lr 1e-4 --epochs 150 --patience 15 \
    --run-name "$rn" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
