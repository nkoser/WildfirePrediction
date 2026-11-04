#!/bin/bash
#SBATCH --job-name=wf_teaser_oracle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Teaser mit Wetter-Oracle: STS-LH, das das beobachtete Wetter der 30 Folgetage kennt (Ablation v31, 3 Seeds).
#   1. Pixel-CIF tau = 1..30 an den drei Teaser-Stichtagen  -> risk_teaser_lho_<datum>.npz (risk_map_oracle.py)
#   2. Vergleich LH / Contagion / Oracle an denselben Pixeln -> metrics/teaser_oracle/ (teaser_oracle_compare.py)
#   3. Teaser-Abbildung des Oracle wie 151                   -> figures/teaser/teaser_v1_lho.png
# Bricht ab, wenn eine Ausgabe schon existiert (nichts ueberschreiben).
# ---------------------------------------------------------------------------
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CK=/data/Wildfire/checkpoints/paper
D="2023-05-19 2023-06-02 2025-05-26"
[ -e "$EXP/figures/teaser/teaser_v1_lho.png" ] && { echo "ABBRUCH: teaser_v1_lho.png existiert"; exit 3; }
"$PY" -m wildfire.paper.risk_map_oracle --config "$CONFIG" --pattern "$CK/sts_cache_G_next_cube_abl_oracle_s?_s?.pt" \
  --dates $D --taus "$(seq -s, 1 30)" --tag teaser_lho || exit 1
"$PY" -m wildfire.paper.teaser_oracle_compare --config "$CONFIG" || exit 1
EV=$("$PY" -c "import json;print(' '.join('--event '+d+':'+','.join(map(str,b)) for d,b in json.load(open('$EXP/figures/teaser/teaser_v1_cgs.json'))['events']))")
"$PY" -m wildfire.paper.teaser_spread --config "$CONFIG" --tag teaser_lho --dates "$D" plot $EV \
  --model-name "STS-LH + weather oracle" --out-name teaser_v1_lho
rc=$?
echo "[$(date -u +%FT%TZ)] exit $rc"; exit $rc
