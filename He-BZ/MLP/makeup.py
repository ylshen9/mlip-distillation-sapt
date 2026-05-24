import os, numpy as np
from pathlib import Path

ROOT = Path("dataset")

for setdir in sorted(ROOT.glob("set.*")):
    print(f"fix {setdir}")

    p = setdir / "coord.npy"
    if p.exists():
        arr = np.load(p)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
            np.save(p, arr)

    p = setdir / "force.npy"
    if p.exists():
        arr = np.load(p)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
            np.save(p, arr)

    p = setdir / "box.npy"
    if p.exists():
        arr = np.load(p)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
            np.save(p, arr)

    p = setdir / "virial.npy"
    if p.exists():
        arr = np.load(p)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
            np.save(p, arr)

print("done")