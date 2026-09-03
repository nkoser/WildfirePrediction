#!/bin/bash
#SBATCH --job-name=wf_tests
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/Wildfire/logs/%x_%j.out
#SBATCH --error=/data/Wildfire/logs/%x_%j.err
#SBATCH --container-image=/raid/data/nik/SR
#SBATCH --container-mounts=/home/nik/Workspace:/workspace,/raid/data/nik:/data,/firedata:/firedata:ro
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-name=wf_tests
# BEWUSST OHNE --partition und ohne --gres: reine CPU-Jobs nehmen so die
# Default-Partition und warten nicht in der GPU-Queue.
#
# Unit- und Eigenschaftstests der Datenpipeline. Exit != 0 bei Fehlschlag,
# damit ein kaputter Stand sichtbar wird statt still weiterzulaufen.
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh
cd "$REPO"
"$PY" -m pytest wildfire/tests -q --tb=short
rc=$?
echo "[$(date -u +%FT%TZ)] pytest exit $rc"
exit $rc
