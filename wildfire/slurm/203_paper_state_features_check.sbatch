#!/bin/bash
#SBATCH --job-name=wf_sfeat_chk
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/shared/nik/Wildfire/logs/%x_%j.out
#SBATCH --error=/scratch/shared/nik/Wildfire/logs/%x_%j.err
# Diagnose der Zustandsmerkmale (state_features/train.npy): Verteilung je Block von Zeilen. Nur lesen.
source /home/nik/workspace/WildfirePrediction/wildfire/slurm/_apptainer.sh
source /workspace/WildfirePrediction/wildfire/slurm/_common.sh || exit 1
"$PY" - <<'PY'
import numpy as np
f = np.load("/data/Wildfire/derived_data/state_features/train.npy", mmap_mode="r")
m = np.load("/data/Wildfire/derived_data/paper_pixel/train_meta.npz")
print("keys meta", list(m.keys()), {k: m[k].shape for k in m})
for a in range(0, f.shape[0], 20000):
    x = np.asarray(f[a:a + 20000]).astype(np.float32)
    zero = (np.abs(x).sum(-1) == 0).mean()
    print(a, "ds_log", x[..., 0].mean().round(4), "none", x[..., 1].mean().round(4), "isO", x[..., 3].mean().round(4),
          "isB", x[..., 4].mean().round(4), "allzero", round(float(zero), 4), "ds_log==0&none==0", round(float(((x[...,0]==0)&(x[...,1]==0)).mean()),4),
          "L range", int(m["L"][a:a+20000].min()), int(m["L"][a:a+20000].max()), "kind", np.bincount(m["kind"][a:a+20000]) if "kind" in m else "")
PY
echo "[$(date -u +%FT%TZ)] exit $?"
