import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from stdplot_tool import mpl_std_Params
from train import BindingEnergyTransformer, load_type_map

try:
    from scipy.special import sph_harm_y as sph_harm
except Exception:
    from scipy.special import sph_harm as sph_harm


NPY_PATH = "../../HeBz_sector_2475.npy"
MODEL_PT = "./model_final.pt"
TYPE_MAP_PATH = "./type_map.raw"
RC_NPZ_PATH = "./rc_fit.npz"

CELL = [20.0, 20.0, 20.0]
FAR_Z = 12.0
CM1_PER_eV = 8065.544
OUT_PNG = "val.png"

D_MODEL = 576
NUM_LAYERS = 8
SOG_N_GAUSSIANS = 12
N_RBF = 28
RC_MIN = 0.8
RC_MAX = 12.0
RC_LR = 12.0

E_ABS_MAX_CM1 = 1000.0
MAX_POINTS = None

BZ_SYMS = ["C"] * 6 + ["H"] * 6
BZ_POS = np.array([
    [-1.2073830, -0.6970829, 0.0],
    [-1.2073830,  0.6970829, 0.0],
    [ 0.0,        1.3941659, 0.0],
    [ 1.2073830,  0.6970829, 0.0],
    [ 1.2073830, -0.6970829, 0.0],
    [ 0.0,       -1.3941659, 0.0],
    [-2.1490090, -1.2407309, 0.0],
    [-2.1490090,  1.2407309, 0.0],
    [ 0.0,        2.4814619, 0.0],
    [ 2.1490090,  1.2407309, 0.0],
    [ 2.1490090, -1.2407309, 0.0],
    [ 0.0,       -2.4814619, 0.0],
], dtype=np.float32)

HE_ATOM_INDEX = len(BZ_SYMS)

arr = np.load(NPY_PATH)
xyz_all = arr[:, :3].astype(np.float32)
Eb_ref_cm1_all = arr[:, 3].astype(np.float64)

mask = np.abs(Eb_ref_cm1_all) < E_ABS_MAX_CM1
xyz = xyz_all[mask]
Eb_ref_cm1 = Eb_ref_cm1_all[mask]

if MAX_POINTS is not None:
    xyz = xyz[:MAX_POINTS]
    Eb_ref_cm1 = Eb_ref_cm1[:MAX_POINTS]

type_list = load_type_map(Path(TYPE_MAP_PATH))
type_dict = {sym: i for i, sym in enumerate(type_list)}

Z_np = np.array([type_dict[s] for s in BZ_SYMS] + [type_dict["He"]], dtype=np.int64)
Z_t = torch.tensor(Z_np, dtype=torch.long).unsqueeze(0)

box_mat = np.array([
    [CELL[0], 0.0, 0.0],
    [0.0, CELL[1], 0.0],
    [0.0, 0.0, CELL[2]],
], dtype=np.float32)

bz_center = BZ_POS.mean(axis=0)


def _sph_eval(m, l, phi, theta):
    if getattr(sph_harm, "__name__", "") == "sph_harm_y":
        return sph_harm(m, l, theta, phi)
    return sph_harm(m, l, phi, theta)


def _sph_real_design(theta, phi, lmax):
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)
    cols = []

    for l in range(lmax + 1):
        Y0 = _sph_eval(0, l, phi, theta)
        cols.append(np.real(Y0))

        for m in range(1, l + 1):
            Y = _sph_eval(m, l, phi, theta)
            cols.append(np.sqrt(2.0) * ((-1) ** m) * np.real(Y))
            cols.append(np.sqrt(2.0) * ((-1) ** m) * np.imag(Y))

    return np.column_stack(cols)


def load_rc_model(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "lmax": int(np.asarray(d["lmax"]).item()),
        "coeff": np.asarray(d["coeff"], float),
    }


rc_model = load_rc_model(RC_NPZ_PATH)


def rc_for_he(he_pos):
    v = np.asarray(he_pos, dtype=np.float32) - bz_center
    r = max(float(np.linalg.norm(v)), 1e-12)

    theta = float(np.arccos(np.clip(v[2] / r, -1.0, 1.0)))
    phi = float(np.mod(np.arctan2(v[1], v[0]), 2.0 * np.pi))

    A = _sph_real_design(np.array([theta]), np.array([phi]), rc_model["lmax"])
    rc = float((A @ rc_model["coeff"]).reshape(-1)[0])

    return max(rc, 1e-3)


model = BindingEnergyTransformer(
    n_types=len(type_list),
    he_index=HE_ATOM_INDEX,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    sog_n_gaussians=SOG_N_GAUSSIANS,
    n_rbf=N_RBF,
    rc_min=RC_MIN,
    rc_max=RC_MAX,
    rc_lr=RC_LR,
)

state = torch.load(MODEL_PT, map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()


def model_energy_eV(he_pos):
    pos = np.vstack([BZ_POS, np.asarray(he_pos, dtype=np.float32)]).astype(np.float32)
    Rc_center = np.array([rc_for_he(he_pos)], dtype=np.float32)

    pos_t = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    box_t = torch.tensor(box_mat, dtype=torch.float32).unsqueeze(0)
    Rc_t = torch.tensor(Rc_center, dtype=torch.float32)

    with torch.no_grad():
        E_total, *_ = model(Z_t, pos_t, box_t, Rc_t)

    return float(E_total.item())


E_far = model_energy_eV([0.0, 0.0, FAR_Z])

Eb_pred_cm1 = np.empty(len(xyz), dtype=np.float64)

for i, (x, y, z) in enumerate(xyz):
    E_tot = model_energy_eV([float(x), float(y), float(z)])
    Eb_pred_cm1[i] = E_tot * CM1_PER_eV

MAE = float(np.mean(np.abs(Eb_pred_cm1 - Eb_ref_cm1)))

mn = float(min(Eb_ref_cm1.min(), Eb_pred_cm1.min()))
mx = float(max(Eb_ref_cm1.max(), Eb_pred_cm1.max()))

mpl_std_Params(0.25, y=1, cmap="Set2")

pad = 0.04 * (mx - mn) if mx > mn else 1.0
xmin = mn - pad

fig, ax = plt.subplots()

ax.scatter(Eb_ref_cm1, Eb_pred_cm1, s=3, alpha=0.8, rasterized=True)
ax.plot([xmin, 1000], [xmin, 1000], linestyle="--", linewidth=0.8, color="black", zorder=1)

ax.set_xlim(xmin, 1000)
ax.set_ylim(xmin, 1000)
ax.set_xlabel(r"$E_{\mathrm{ref}}$ (cm$^{-1}$)")
ax.set_ylabel(r"$E_{\mathrm{predict}}$ (cm$^{-1}$)")
ax.set_aspect("equal", adjustable="box")
ax.tick_params(top=False, right=False)
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))

neg_mask = (Eb_ref_cm1 < 0) & (Eb_pred_cm1 < 0)

if np.any(neg_mask):
    x_neg = Eb_ref_cm1[neg_mask]
    y_neg = Eb_pred_cm1[neg_mask]

    neg_min = float(min(x_neg.min(), y_neg.min()))
    neg_max = float(max(x_neg.max(), y_neg.max()))

    neg_pad = 0.08 * (neg_max - neg_min) if neg_max > neg_min else 5.0
    zx1 = neg_min - neg_pad
    zx2 = min(0.0, neg_max + neg_pad)

    axins = ax.inset_axes([0.66, 0.12, 0.30, 0.30])
    axins.scatter(x_neg, y_neg, s=3, alpha=0.8, rasterized=True)
    axins.plot([zx1, zx2], [zx1, zx2], linestyle="--", linewidth=0.8, color="black", zorder=1)

    axins.set_xlim(zx1, zx2)
    axins.set_ylim(zx1, zx2)
    axins.set_aspect("equal", adjustable="box")
    axins.set_xticks([-50, 0])
    axins.set_yticks([-50, 0])
    axins.tick_params(top=False, right=False, labelsize=7)

    for spine in axins.spines.values():
        spine.set_linewidth(ax.spines["bottom"].get_linewidth())

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=True)
plt.savefig(Path(OUT_PNG).with_suffix(".svg"), dpi=600, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"MAE = {MAE:.6f} cm^-1")