# Wird als ERSTE Zeile jedes sbatch-Skripts gesourced, noch auf dem HOST.
#
# Startet das aufrufende Skript neu - diesmal im Apptainer-Image - und kehrt
# nicht zurueck. Im Container ist $APPTAINER_CONTAINER gesetzt, beim zweiten
# Durchlauf tut diese Datei deshalb nichts, und das Skript laeuft ab der
# naechsten Zeile normal weiter.
#
# Die Binds bilden die Container-Pfade des alten Clusters nach. Python-Code,
# configs/audit.yaml und die Manifeste (d4_cube.json speichert absolute Pfade)
# bleiben dadurch unveraendert gueltig:
#
#   HOST                                  CONTAINER
#   /home/nik/workspace/WildfirePrediction  /workspace/WildfirePrediction
#   /scratch/shared/nik/Wildfire            /data/Wildfire          (EXPERIMENT_ROOT)
#   /cold/Niklas/FireData                   /firedata  :ro          (Rohdaten)
#
# Warum eine Kopie des Skripts: sbatch fuehrt eine Kopie unter
# /var/spool/slurmd aus, die im Container nicht sichtbar ist. Sie wird nach
# $EXP/logs/jobscripts gelegt - bleibt so auch als Provenienz liegen.
#
# Uebersteuerbar:  WF_SIF, WF_HOST_REPO, WF_HOST_EXP, WF_HOST_RAW
#                  WF_EXTRA_BINDS="/quelle:/ziel[,...]"
#
# --output/--error im #SBATCH-Kopf werden vom HOST aufgeloest, BEVOR dieses
# Skript laeuft: dort muss der Host-Pfad /scratch/shared/nik/Wildfire/logs
# stehen, und das Verzeichnis muss vor dem Submit existieren.

if [ -z "${APPTAINER_CONTAINER:-}" ]; then
    WF_HOST_REPO="${WF_HOST_REPO:-/home/nik/workspace/WildfirePrediction}"
    WF_HOST_EXP="${WF_HOST_EXP:-/scratch/shared/${USER}/Wildfire}"
    WF_HOST_RAW="${WF_HOST_RAW:-/cold/Niklas/FireData}"
    WF_SIF="${WF_SIF:-/scratch/shared/${USER}/containers/wildfire.sif}"

    if [ ! -f "$WF_SIF" ]; then
        echo "ABBRUCH: Image $WF_SIF fehlt - zuerst sbatch wildfire/slurm/00a_build_container.sbatch" >&2
        exit 3
    fi
    mkdir -p "$WF_HOST_EXP/logs/jobscripts"
    _wf_name="${SLURM_JOB_ID:-$$}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}.sh"
    cp "$0" "$WF_HOST_EXP/logs/jobscripts/$_wf_name"

    _wf_binds=(--bind "$WF_HOST_REPO:/workspace/WildfirePrediction"
               --bind "$WF_HOST_EXP:/data/Wildfire"
               --bind "$WF_HOST_RAW:/firedata:ro")
    [ -n "${WF_EXTRA_BINDS:-}" ] && _wf_binds+=(--bind "$WF_EXTRA_BINDS")
    # --nv nur mit zugewiesener GPU; ohne gibt es sonst eine Warnung je Job.
    _wf_nv=(); [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && _wf_nv=(--nv)

    echo "[apptainer] $WF_SIF  job=${SLURM_JOB_ID:-?}  host=$(hostname)"
    exec apptainer exec "${_wf_nv[@]}" "${_wf_binds[@]}" \
        --pwd /workspace/WildfirePrediction "$WF_SIF" \
        bash "/data/Wildfire/logs/jobscripts/$_wf_name" "$@"
fi
