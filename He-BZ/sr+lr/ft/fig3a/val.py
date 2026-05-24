import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from stdplot_tool import mpl_std_Params

from train import BindingEnergyTransformer, load_type_map

NPY_PATH = "./validation.npy"
MODEL_PT = "./model_final.pt"
TYPE_MAP_PATH = "../dataset/type_map.raw"

CELL = [20.0, 20.0, 20.0]
CM1_PER_eV = 8065.544

OUT_PNG = "lt1000.png"
OUT_SVG = "lt1000.svg"

E_ABS_MAX_CM1 = 1000.0

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
], dtype=np.float64)

HE_ATOM_INDEX = len(BZ_SYMS)

arr = np.load(NPY_PATH)
xyz_all = arr[:, :3].astype(np.float64)
Eb_ref_cm1_all = arr[:, 3].astype(np.float64)

mask = np.abs(Eb_ref_cm1_all) < E_ABS_MAX_CM1
xyz = xyz_all[mask]
Eb_ref_cm1 = Eb_ref_cm1_all[mask]

type_list = load_type_map(Path(TYPE_MAP_PATH))
type_dict = {sym: i for i, sym in enumerate(type_list)}

model = BindingEnergyTransformer(
    n_types=len(type_list),
    he_index=HE_ATOM_INDEX,
    d_model=576,
    nhead=12,
    num_layers=8,
    use_long_range=True,
    sog_n_gaussians=12,
    sog_r_switch=4.18,
    sog_r_max=12.0,
    n_rbf=28,
)

state = torch.load(MODEL_PT, map_location="cpu")
torch.nn.Module.load_state_dict(model, state, strict=True)
model.eval()

BZ_Z = np.array([type_dict[s] for s in BZ_SYMS], dtype=np.int64)
He_Z = np.array([type_dict["He"]], dtype=np.int64)
Z_all = np.concatenate([BZ_Z, He_Z])

box_mat = np.array([
    [CELL[0], 0.0, 0.0],
    [0.0, CELL[1], 0.0],
    [0.0, 0.0, CELL[2]],
], dtype=np.float32)


def model_energy(he_pos):
    pos = np.vstack([BZ_POS, np.asarray(he_pos, dtype=np.float64)])
    Z = torch.tensor(Z_all, dtype=torch.long).unsqueeze(0)
    pos_t = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    box = torch.tensor(box_mat, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        E_total, *_ = model(Z, pos_t, box)

    return float(E_total.item())


Eb_pred_eV = np.array([model_energy(p) for p in xyz])
Eb_pred_cm1 = Eb_pred_eV * CM1_PER_eV

MAE = np.mean(np.abs(Eb_pred_cm1 - Eb_ref_cm1))
print(f"MAE = {MAE:.6f} cm^-1")

mpl_std_Params(0.25, y=1, cmap="Set2")

mn = float(min(Eb_ref_cm1.min(), Eb_pred_cm1.min()))
mx = float(max(Eb_ref_cm1.max(), Eb_pred_cm1.max()))
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
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
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
plt.savefig(OUT_SVG, dpi=600, bbox_inches="tight", transparent=True)
plt.close(fig)
