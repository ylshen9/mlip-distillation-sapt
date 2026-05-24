import os
import numpy as np
from pathlib import Path

NPY_PATH = "BZ-DFT.npy"
OUT_ROOT = "dataset"
CELL_A = 20.0
CM1_PER_eV = 8065.544
SEED = 42
TRAIN_RATIO = 0.8

BZ_SYMS = ["C"] * 6 + ["H"] * 6
BZ_POS = np.array([
    [-1.2073830, -0.6970829, 0.0],
    [-1.2073830,  0.6970829, 0.0],
    [ 0.0000000,  1.3941659, 0.0],
    [ 1.2073830,  0.6970829, 0.0],
    [ 1.2073830, -0.6970829, 0.0],
    [ 0.0000000, -1.3941659, 0.0],
    [-2.1490090, -1.2407309, 0.0],
    [-2.1490090,  1.2407309, 0.0],
    [ 0.0000000,  2.4814619, 0.0],
    [ 2.1490090,  1.2407309, 0.0],
    [ 2.1490090, -1.2407309, 0.0],
    [ 0.0000000, -2.4814619, 0.0],
], dtype=float)

TYPE_MAP = ["C", "H", "He"]
type_idx = {"C": 0, "H": 1, "He": 2}

SYMS = BZ_SYMS + ["He"]
POS0 = BZ_POS

arr = np.load(NPY_PATH)
if arr.shape[1] < 4:
    raise ValueError("NPY must have at least 4 columns: x y z Eb_ref_cm^-1")

xyz = arr[:, :3].astype(float)
Eb_ref_cm1 = arr[:, 3].astype(float)

mask = np.abs(Eb_ref_cm1) < 1000.0
xyz = xyz[mask]
Eb_ref_cm1 = Eb_ref_cm1[mask]

if len(xyz) == 0:
    raise RuntimeError("No samples after masking (Eb_ref_cm^-1 < 1000). Check your data.")

rng = np.random.default_rng(SEED)
n_all = len(xyz)
perm_all = rng.permutation(n_all)
n_use = 3200
use_sel = perm_all[:n_use]

xyz = xyz[use_sel]
Eb_ref_cm1 = Eb_ref_cm1[use_sel]

Eb_ref_eV = Eb_ref_cm1 / CM1_PER_eV

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(xyz))

n_train = 2400
train_sel = perm[:n_train]
val_sel = perm[n_train:]

xyz_train = xyz[train_sel]
xyz_val = xyz[val_sel]
E_train = Eb_ref_eV[train_sel]
E_val = Eb_ref_eV[val_sel]

print(f"[INFO] After mask (<1000 cm^-1): total={len(xyz)}  train={len(xyz_train)}  val={len(xyz_val)}")
print(f"[INFO] Energy (eV) stats -> train: min={E_train.min():.6f}, max={E_train.max():.6f} ; val: min={E_val.min():.6f}, max={E_val.max():.6f}")

def write_type_raw_text(path: Path, types):
    with open(path, "w") as f:
        for t in types:
            f.write(f"{t}\n")

def write_system(subdir: Path, xyz_subset: np.ndarray, energy_subset: np.ndarray):
    nat = len(SYMS)
    nframes = len(xyz_subset)

    coord = np.zeros((nframes, nat * 3), dtype=np.float32)
    box = np.zeros((nframes, 9), dtype=np.float32)
    ener = np.array(energy_subset, dtype=np.float64)

    box_single = np.diag([CELL_A, CELL_A, CELL_A]).reshape(-1)

    for i, (x, y, z) in enumerate(xyz_subset):
        pos = np.vstack([POS0, [x, y, z]])
        coord[i, :] = pos.reshape(-1)
        box[i, :] = box_single

    subdir.mkdir(parents=True, exist_ok=True)
    np.save(subdir / "coord.npy", coord)
    np.save(subdir / "box.npy", box)
    np.save(subdir / "energy.npy", ener)

    atom_types = [type_idx[s] for s in SYMS]
    write_type_raw_text(subdir / "type.raw", atom_types)

root = Path(OUT_ROOT)
root.mkdir(parents=True, exist_ok=True)

with open(root / "type_map.raw", "w") as f:
    for s in TYPE_MAP:
        f.write(f"{s}\n")

write_system(root / "set.000", xyz_train, E_train)
write_system(root / "set.001", xyz_val, E_val)

print(f"[OK] Dataset generated under: {OUT_ROOT}")
print(f"  set.000 (train): {len(xyz_train)} frames")
print(f"  set.001 (val):   {len(xyz_val)} frames")