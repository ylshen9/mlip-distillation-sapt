import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase import Atoms

try:
    from deepmd.calculator import DP
except Exception as e:
    raise RuntimeError(
        "Failed to import deepmd.calculator.DP. Please install deepmd-kit."
    ) from e

NPY_PATH = "../../HeBz_sector_2475.npy"
N_USE = None
CELL, PBC = [20, 20, 20], False
FAR_Z = 12.0
CM1_PER_eV = 8065.544
OUT_PNG = "val.png"

MODEL_PATH = "./model_final.pb"

DATASET_ROOT = Path("dataset")
E_ABS_MAX_CM1 = 1000.0
ROUND_DECIMALS = 6
MAX_HOLDOUT = None

FALLBACK_TYPE_ORDER = ["C", "H", "He"]

BZ_SYMS = ["C"] * 6 + ["H"] * 6
BZ_POS = np.array(
    [
        [-1.2073830, -0.6970829, 0.0],
        [-1.2073830,  0.6970829, 0.0],
        [0.0, 1.3941659, 0.0],
        [1.2073830,  0.6970829, 0.0],
        [1.2073830, -0.6970829, 0.0],
        [0.0, -1.3941659, 0.0],
        [-2.1490090, -1.2407309, 0.0],
        [-2.1490090,  1.2407309, 0.0],
        [0.0, 2.4814619, 0.0],
        [2.1490090,  1.2407309, 0.0],
        [2.1490090, -1.2407309, 0.0],
        [0.0, -2.4814619, 0.0],
    ],
    dtype=np.float32,
)

N_ATOMS = len(BZ_SYMS) + 1


def _xyz_keys(xyz: np.ndarray, decimals: int):
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz_r = np.round(xyz, decimals=decimals)
    return [tuple(row.tolist()) for row in xyz_r]


def load_used_he_xyz_from_dataset(dataset_root: Path) -> np.ndarray:
    used_list = []
    for sub in ("set.000", "set.001"):
        coord_path = dataset_root / sub / "coord.npy"
        if not coord_path.exists():
            continue
        coord = np.load(coord_path)
        if coord.ndim != 2 or coord.shape[1] != N_ATOMS * 3:
            raise ValueError(
                f"Unexpected coord.npy shape in {coord_path}: {coord.shape}"
            )
        coord = coord.reshape(coord.shape[0], N_ATOMS, 3)
        he_xyz = coord[:, -1, :]
        used_list.append(he_xyz.astype(np.float64))

    if not used_list:
        raise FileNotFoundError(f"No coord.npy found under {dataset_root}")

    return np.vstack(used_list)


def build_holdout_from_raw(raw_xyz: np.ndarray, raw_Ecm1: np.ndarray, used_he_xyz: np.ndarray):
    eligible_mask = np.abs(raw_Ecm1) < E_ABS_MAX_CM1
    eligible_idx = np.where(eligible_mask)[0]

    used_keys = set(_xyz_keys(used_he_xyz, decimals=ROUND_DECIMALS))
    eligible_keys = _xyz_keys(raw_xyz[eligible_idx], decimals=ROUND_DECIMALS)

    keep_mask = np.array([k not in used_keys for k in eligible_keys], dtype=bool)
    holdout_idx = eligible_idx[keep_mask]

    stats = {
        "eligible": int(len(eligible_idx)),
        "used_trainval": int(len(used_he_xyz)),
        "holdout": int(len(holdout_idx)),
        "removed": int(len(eligible_idx) - len(holdout_idx)),
    }
    return raw_xyz[holdout_idx], raw_Ecm1[holdout_idx], stats


arr = np.load(NPY_PATH)
if N_USE is not None and N_USE < len(arr):
    arr = arr[:N_USE]

raw_xyz = arr[:, :3].astype(np.float64)
raw_Ecm1 = arr[:, 3].astype(np.float64)

used_he_xyz = load_used_he_xyz_from_dataset(DATASET_ROOT)
xyz, Eb_ref_cm1_all, stats = build_holdout_from_raw(raw_xyz, raw_Ecm1, used_he_xyz)

print(f"eligible = {stats['eligible']}")
print(f"used_trainval = {stats['used_trainval']}")
print(f"holdout = {stats['holdout']}")
print(f"removed = {stats['removed']}")

if MAX_HOLDOUT is not None and len(xyz) > MAX_HOLDOUT:
    xyz = xyz[:MAX_HOLDOUT]
    Eb_ref_cm1_all = Eb_ref_cm1_all[:MAX_HOLDOUT]


def load_type_order_from_typemap(model_path: str):
    cand = []
    model_dir = Path(model_path).resolve().parent
    cand.append(model_dir / "type_map.raw")
    cand.append(Path.cwd() / "type_map.raw")
    cand.append(DATASET_ROOT / "type_map.raw")
    for p in cand:
        if p.is_file():
            with open(p, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if len(lines) == 1 and " " in lines[0]:
                order = [tok for tok in lines[0].split() if tok]
            else:
                order = lines
            return order
    return FALLBACK_TYPE_ORDER


TYPE_ORDER = load_type_order_from_typemap(MODEL_PATH)
TYPE_DICT = {sym: i for i, sym in enumerate(TYPE_ORDER)}


def make_dp_calculator(model_path: str, type_dict: dict):
    return DP(model=model_path, type_dict=type_dict)


calc = make_dp_calculator(MODEL_PATH, TYPE_DICT)


def Etot_he_bz(he_pos):
    atoms = Atoms(
        BZ_SYMS + ["He"],
        positions=np.vstack([BZ_POS, he_pos]),
        cell=CELL,
        pbc=PBC,
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = calc
    return atoms.get_potential_energy()


E_far = Etot_he_bz([0.0, 0.0, FAR_Z])

Eb_pred_eV_all = np.array([Etot_he_bz([x, y, z]) - E_far for x, y, z in xyz], dtype=float)
Eb_pred_cm1_all = Eb_pred_eV_all * CM1_PER_eV

diff_all = Eb_pred_cm1_all - Eb_ref_cm1_all
MAE_holdout_cm1 = float(np.mean(np.abs(diff_all)))

n_plot = len(xyz)
if n_plot == 0:
    raise ValueError("holdout empty")

mn = float(min(Eb_ref_cm1_all.min(), Eb_pred_cm1_all.min()))
mx = float(max(Eb_ref_cm1_all.max(), Eb_pred_cm1_all.max()))

plt.figure(figsize=(7, 5.5))
plt.scatter(Eb_ref_cm1_all, Eb_pred_cm1_all, s=18, alpha=0.85)
plt.plot([mn, mx], [mn, mx], lw=1.2)

title = (
    f"He@benzene binding HOLDOUT (n={n_plot})\n"
    f"MAE = {MAE_holdout_cm1:.1f} cm$^{{-1}}$"
)
plt.title(title)
plt.xlabel("Reference (cm$^{-1}$)")
plt.ylabel("Predicted (cm$^{-1}$)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=220)
plt.show()

print(f"Saved figure -> {OUT_PNG}")
print(f"MODEL_PATH={MODEL_PATH}")
print(f"TYPE_ORDER={TYPE_ORDER}")
print(f"TYPE_DICT={TYPE_DICT}")
print(f"n_raw={len(raw_xyz)} eligible={stats['eligible']} used={stats['used_trainval']} holdout={stats['holdout']}")
print(f"HOLDOUT MAE = {MAE_holdout_cm1:.6f} cm^-1")