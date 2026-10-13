#!/bin/bash
#SBATCH --job-name=wf_tests
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# BEWUSST OHNE --partition und ohne --gres: reine CPU-Jobs nehmen so die
# Default-Partition und warten nicht in der GPU-Queue.
#
# Unit- und Eigenschaftstests der Datenpipeline. Exit != 0 bei Fehlschlag,
# damit ein kaputter Stand sichtbar wird statt still weiterzulaufen.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || {
    echo "ABBRUCH: _common.sh nicht gefunden - laeuft der Container ueberhaupt?" >&2
    exit 1
}
cd "$REPO"
"$PY" -m pytest wildfire/tests -q --tb=short
rc=$?
echo "[$(date -u +%FT%TZ)] pytest exit $rc"
exit $rc
