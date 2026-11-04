#!/bin/bash
#SBATCH --job-name=wf_sx_episode
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Contagion-STS mit Brenndauer ab Episodenbeginn (wildfire/paper/sts_episode.py, contagion_episode.py),
# sonst wie cg_ref aus 166 (Kern 7, 30 Beobachtungstage, Brenndauer-Kurve, ohne Warmstart, ohne Zusatz).
#   0-2   cg_episode   Seeds 0..2, Luecke EP_GAP (Standard 2) Tage
# Ausgaben NEU: checkpoints/paper/sts_extreme_cg_episode<SFX>_s<seed>_s<seed>.pt, p35_sts_extreme_..., p47_sts_episode_...
# Rauchtest: EXTRA="--epochs 1 --steps-per-epoch 20 --limit-train 2000 --limit-eval 500" SFX=_smoke
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
seed=${SLURM_ARRAY_TASK_ID:?Array-Job erforderlich}
n=cg_episode${SFX:-}
if [[ -e "$EXP/data_manifests/p47_sts_episode_${n}_s${seed}.json" ]]; then
  echo "ABBRUCH: Manifest p47_sts_episode_${n}_s${seed}.json existiert schon"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m wildfire.paper.sts_episode --config "$CONFIG" --ep-gap "${EP_GAP:-2}" \
    --head contagion --cg-kernel 7 --cg-obs-days 30 --cg-burn curve \
    --seeds $seed --tag "_${n}_s${seed}" ${EXTRA:-}
rc=$?   # Exit-Code des Python-Aufrufs weitergeben, sonst greift afterok nicht
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
