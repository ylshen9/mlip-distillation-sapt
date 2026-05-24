import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except Exception:
    PchipInterpolator = None
    HAS_SCIPY = False

from train import BindingEnergyTransformer
from stdplot_tool import mpl_std_Params


CM1_PER_eV = 8065.544

MODEL_ORB = Path("model_orb.pt")
MODEL_MS = Path("model_mattersim.pt")

CORONENE_NPY = Path("../../coronene.npy")
MULTI_NPY = Path("../../multi.npy")

CELL_A = 40.0

XY_TOL = 1e-6
Z_MIN = 0.0
Z_MAX = 10.0

OUT_DIR = Path("combined")
OUT_DIR.mkdir(exist_ok=True)

MULTI_METHOD = "pbe0_def2-SVP"

XY_TARGETS = {
    "Coronene": (2.0, 0.0),
    "C_Coronene": (2.0, 0.0),
    "CC_Coronene": (2.0, 0.0),
    "CCC_Coronene": (2.0, 0.0),
}

CORONENE_SYMS = ["C"] * 24 + ["H"] * 12
CORONENE_POS = np.array([
    [1.211568, 3.492552, 0.0], [2.418855, 2.795525, 0.0],
    [2.424434, 1.399748, 0.0], [3.630423, 0.697027, 0.0],
    [3.630423, -0.697027, 0.0], [2.424434, -1.399748, 0.0],
    [2.418855, -2.795525, 0.0], [1.211568, -3.492552, 0.0],
    [-0.000000, -2.799496, 0.0], [-1.211568, -3.492552, 0.0],
    [-2.418855, -2.795525, 0.0], [-2.424434, -1.399748, 0.0],
    [-3.630423, -0.697027, 0.0], [-3.630423, 0.697027, 0.0],
    [-2.424434, 1.399748, 0.0], [-2.418855, 2.795525, 0.0],
    [-1.211568, 3.492552, 0.0], [-0.000000, 2.799496, 0.0],
    [-0.000000, 1.399577, 0.0], [-1.212069, 0.699788, 0.0],
    [-1.212069, -0.699788, 0.0], [-0.000000, -1.399577, 0.0],
    [1.212069, -0.699788, 0.0], [1.212069, 0.699788, 0.0],
    [1.220986, 4.580106, 0.0], [3.355996, 3.347458, 0.0],
    [4.576981, 1.232649, 0.0], [4.576981, -1.232649, 0.0],
    [3.355996, -3.347458, 0.0], [1.220986, -4.580106, 0.0],
    [-1.220986, -4.580106, 0.0], [-3.355996, -3.347458, 0.0],
    [-4.576981, -1.232649, 0.0], [-4.576981, 1.232649, 0.0],
    [-3.355996, 3.347458, 0.0], [-1.220986, 4.580106, 0.0],
], dtype=float)

BZ_POS = np.array([
    [-1.2073830, -0.6970829, 0.0], [-1.2073830, 0.6970829, 0.0],
    [0.0, 1.3941659, 0.0], [1.2073830, 0.6970829, 0.0],
    [1.2073830, -0.6970829, 0.0], [0.0, -1.3941659, 0.0],
    [-2.1490090, -1.2407309, 0.0], [-2.1490090, 1.2407309, 0.0],
    [0.0, 2.4814619, 0.0], [2.1490090, 1.2407309, 0.0],
    [2.1490090, -1.2407309, 0.0], [0.0, -2.4814619, 0.0],
], dtype=float)

BZ_C = BZ_POS[:6, :2]
BZ_H = BZ_POS[6:, :2]
R_C = np.linalg.norm(BZ_C, axis=1).mean()
R_H = np.linalg.norm(BZ_H, axis=1).mean()
R_CH = R_H - R_C
L = R_C * math.sqrt(3.0)

a1 = np.array([L, 0.0])
a2 = np.array([0.5 * L, 0.5 * math.sqrt(3.0) * L])


def make_hex(n_ring):
    centers = []

    for u in range(-n_ring, n_ring + 1):
        for v in range(-n_ring, n_ring + 1):
            w = -u - v
            if max(abs(u), abs(v), abs(w)) <= n_ring:
                centers.append((u, v))

    centers_cart = np.array([u * a1 + v * a2 for u, v in centers], dtype=float)

    C_list = []
    for cx, cy in centers_cart:
        for k in range(6):
            ang = math.pi / 2.0 + k * math.pi / 3.0
            C_list.append([cx + R_C * math.cos(ang), cy + R_C * math.sin(ang)])

    def dedup(pts):
        tbl = {}
        for x, y in pts:
            tbl[(round(x, 4), round(y, 4))] = [x, y]
        return np.array(list(tbl.values()), dtype=float)

    C_2d = dedup(C_list)
    N = len(C_2d)

    neighbors = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(C_2d[i] - C_2d[j])
            if 0.8 * R_C < d < 1.2 * R_C:
                neighbors[i].append(j)
                neighbors[j].append(i)

    H_list = []
    for i in range(N):
        if len(neighbors[i]) < 3:
            v_sum = sum(C_2d[j] - C_2d[i] for j in neighbors[i])
            out = -v_sum
            nm = np.linalg.norm(out)
            if nm > 0:
                H_list.append(C_2d[i] + out / nm * R_CH)

    H_2d = np.array(H_list, dtype=float)

    C_3d = np.column_stack([C_2d, np.zeros(len(C_2d))])
    H_3d = np.column_stack([H_2d, np.zeros(len(H_2d))])

    return ["C"] * len(C_3d) + ["H"] * len(H_3d), np.vstack([C_3d, H_3d])


def to_xyze_cm1(arr):
    arr = np.asarray(arr, dtype=float)
    return arr[:, :3], arr[:, 3]


def load_npy_object(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    return data


def get_array_from_data(data, key=None, method=None):
    if not isinstance(data, dict):
        return data

    aliases = {
        "Coronene": ["Coronene", "coronene", "Cor"],
        "C_Coronene": ["C_Coronene", "C-Coronene", "C coronene", "C"],
        "CC_Coronene": ["CC_Coronene", "CC-Coronene", "CC coronene", "CC"],
        "CCC_Coronene": ["CCC_Coronene", "CCC-Coronene", "CCC coronene", "CCC"],
    }

    candidates = aliases.get(key, [key])

    for candidate in candidates:
        if candidate in data:
            obj = data[candidate]
            if isinstance(obj, dict) and method is not None and method in obj:
                return obj[method]
            return obj

    if method is not None and method in data:
        return data[method]

    raise KeyError(f"Cannot find {key}. Available keys: {list(data.keys())}")


def select_xy_line_points(xyz, E, x_target, y_target, use_nearest=False):
    z_mask = (xyz[:, 2] >= Z_MIN) & (xyz[:, 2] <= Z_MAX)
    xyz_z = xyz[z_mask]
    E_z = E[z_mask]

    if not use_nearest:
        mask = (
            (np.abs(xyz_z[:, 0] - x_target) < XY_TOL)
            & (np.abs(xyz_z[:, 1] - y_target) < XY_TOL)
        )
        idx = np.where(mask)[0]

        if len(idx) > 0:
            idx = idx[np.argsort(xyz_z[idx, 2])]
            return xyz_z[idx], E_z[idx], xyz_z[idx[0], :2]

    xy_unique = np.unique(np.round(xyz_z[:, :2], decimals=6), axis=0)
    dist = np.sqrt((xy_unique[:, 0] - x_target) ** 2 + (xy_unique[:, 1] - y_target) ** 2)
    xy_pick = xy_unique[np.argmin(dist)]

    mask = (
        (np.abs(xyz_z[:, 0] - xy_pick[0]) < 5e-6)
        & (np.abs(xyz_z[:, 1] - xy_pick[1]) < 5e-6)
    )

    idx = np.where(mask)[0]
    idx = idx[np.argsort(xyz_z[idx, 2])]

    return xyz_z[idx], E_z[idx], xy_pick


@torch.no_grad()
def predict_eV(model, Z_1d, host_pos, xyz, box):
    M = xyz.shape[0]
    N = host_pos.shape[0] + 1
    out = np.zeros(M, dtype=float)

    Z = torch.tensor(Z_1d, dtype=torch.long)
    box_t = torch.tensor(box, dtype=torch.float32).unsqueeze(0)

    for i in range(M):
        pos = np.zeros((1, N, 3), dtype=float)
        pos[0, :-1] = host_pos
        pos[0, -1] = xyz[i]

        E, *_ = model(
            Z.unsqueeze(0),
            torch.tensor(pos, dtype=torch.float32),
            box_t,
        )

        out[i] = float(E.item())

    return out


def load_model(path):
    model = BindingEnergyTransformer(
        n_types=3,
        he_type_id=2,
        d_model=576,
        nhead=12,
        num_layers=8,
        use_long_range=True,
        sog_n_gaussians=12,
        sog_r_switch=4.18,
        sog_r_max=12.0,
        n_rbf=28,
    )

    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model.eval()

    return model


def get_host(system_name):
    if system_name == "Coronene":
        return CORONENE_SYMS, CORONENE_POS

    ring_map = {
        "C_Coronene": 2,
        "CC_Coronene": 3,
        "CCC_Coronene": 4,
    }

    return make_hex(ring_map[system_name])


def calc_mae(E_ref, E_pred):
    return float(np.mean(np.abs(E_pred - E_ref)))




def make_smooth_curve(x, y, n=250):
    """Return a dense 1D interpolation curve sorted by x.

    Uses monotone PCHIP interpolation when scipy is available; otherwise
    falls back to numpy linear interpolation. PCHIP avoids most overshoot
    artifacts while making sparse DFT reference points visually smooth.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]

    if len(xu) < 2:
        return xu, yu

    xd = np.linspace(float(xu.min()), float(xu.max()), n)

    if HAS_SCIPY and len(xu) >= 3:
        yd = PchipInterpolator(xu, yu)(xd)
    else:
        yd = np.interp(xd, xu, yu)

    return xd, yd


def predict_line_for_xy(model, name, z_values, xy, type_dict, box):
    """Predict model energy along a generated vertical line at fixed (x,y)."""
    host_syms, host_pos = get_host(name)
    Z = np.array([type_dict[s] for s in host_syms] + [type_dict["He"]], dtype=int)
    xyz = np.column_stack([
        np.full_like(z_values, float(xy[0]), dtype=float),
        np.full_like(z_values, float(xy[1]), dtype=float),
        np.asarray(z_values, dtype=float),
    ])
    E_pred = predict_eV(model, Z, host_pos, xyz, box) * CM1_PER_eV
    return xyz, E_pred


def main():
    type_map = ["C", "H", "He"]
    type_dict = {s: i for i, s in enumerate(type_map)}
    box = np.diag([CELL_A] * 3).astype(float)

    model_orb = load_model(MODEL_ORB)
    model_ms = load_model(MODEL_MS)

    coronene_data = load_npy_object(CORONENE_NPY)
    multi_data = load_npy_object(MULTI_NPY)

    systems = {
        "Coronene": get_array_from_data(coronene_data, "Coronene", method=MULTI_METHOD),
        "C_Coronene": get_array_from_data(multi_data, "C_Coronene", method=MULTI_METHOD),
        "CC_Coronene": get_array_from_data(multi_data, "CC_Coronene", method=MULTI_METHOD),
        "CCC_Coronene": get_array_from_data(multi_data, "CCC_Coronene", method=MULTI_METHOD),
    }

    colors = {
        "Coronene": "red",
        "C_Coronene": "green",
        "CC_Coronene": "orange",
        "CCC_Coronene": "blue",
    }

    short_labels = {
        "Coronene": "Cor",
        "C_Coronene": "C",
        "CC_Coronene": "CC",
        "CCC_Coronene": "CCC",
    }

    ordered_names = ["Coronene", "C_Coronene", "CC_Coronene", "CCC_Coronene"]

    selected = {}

    for name in ordered_names:
        xyz_all, E_all = to_xyze_cm1(systems[name])
        x_target, y_target = XY_TARGETS[name]

        use_nearest = name == "Coronene"

        xyz_line, E_line, xy_used = select_xy_line_points(
            xyz_all,
            E_all,
            x_target=x_target,
            y_target=y_target,
            use_nearest=use_nearest,
        )

        selected[name] = {
            "xyz": xyz_line,
            "z": xyz_line[:, 2],
            "E_ref": E_line,
            "xy": xy_used,
        }

        print(
            f"{name}: target=({x_target:.6f}, {y_target:.6f}), "
            f"used=({xy_used[0]:.6f}, {xy_used[1]:.6f}), "
            f"n={len(xyz_line)}"
        )

    z_scale_names = ["C_Coronene", "CC_Coronene", "CCC_Coronene"]
    common_zmin = min(float(selected[name]["z"].min()) for name in z_scale_names)

    for name in ordered_names:
        z = selected[name]["z"]
        keep = z >= common_zmin

        selected[name]["xyz"] = selected[name]["xyz"][keep]
        selected[name]["z"] = selected[name]["z"][keep]
        selected[name]["E_ref"] = selected[name]["E_ref"][keep]

        print(
            f"{name}: kept z >= {common_zmin:.6f}, "
            f"n={len(selected[name]['z'])}"
        )

    common_zmax = max(float(selected[name]["z"].max()) for name in ordered_names)

    mpl_std_Params(0.17, y=1, cmap="Set2")

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(7, 3),
        sharex="col",
    )

    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8

    def plot_ref_pred(ax, z, E_ref, E_pred, color, show_legend=False):
        z_ref_dense, E_ref_dense = make_smooth_curve(z, E_ref, n=250)
        ax.plot(
            z_ref_dense,
            E_ref_dense,
            linestyle="--",
            linewidth=0.9,
            color=color,
        )
        ax.plot(
            z,
            E_ref,
            linestyle="None",
            marker="o",
            markersize=2.6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.6,
            color=color,
            zorder=5,
        )

        z_pred_dense, E_pred_dense = make_smooth_curve(z, E_pred, n=250)
        ax.plot(
            z_pred_dense,
            E_pred_dense,
            linestyle="-",
            linewidth=1.0,
            color=color,
        )
        ax.plot(
            z,
            E_pred,
            linestyle="None",
            marker="s",
            markersize=2.4,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.3,
            color=color,
            zorder=6,
        )

        ax.tick_params(top=False, right=False)

        if show_legend:
            handles = [
                Line2D(
                    [0], [0],
                    color=color,
                    linestyle="--",
                    linewidth=0.9,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=0.7,
                    label="DFT",
                ),
                Line2D(
                    [0], [0],
                    color=color,
                    linestyle="-",
                    linewidth=1.0,
                    marker="s",
                    markersize=3.8,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.3,
                    label="Predict",
                ),
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                loc="upper right",
                fontsize=6.5,
                handlelength=1.6,
                borderpad=0.2,
                labelspacing=0.2,
            )

    pred_orb_all = {}
    pred_ms_all = {}

    mae_table = {
        "Orb": {},
        "MatterSim": {},
    }

    for col, name in enumerate(ordered_names):
        host_syms, host_pos = get_host(name)
        Z = np.array([type_dict[s] for s in host_syms] + [type_dict["He"]], dtype=int)

        xyz = selected[name]["xyz"]
        z = selected[name]["z"]
        E_ref = selected[name]["E_ref"]

        E_pred = predict_eV(model_orb, Z, host_pos, xyz, box) * CM1_PER_eV
        pred_orb_all[name] = E_pred

        plot_ref_pred(axes[0, col], z, E_ref, E_pred, colors[name], show_legend=False)

        n_atoms = len(host_syms) + 1
        mae = calc_mae(E_ref, E_pred)
        mae_per_atom = mae / n_atoms

        mae_table["Orb"][name] = (mae, mae_per_atom, n_atoms)

    cor_xmin = float(selected["Coronene"]["z"].min())
    cor_xmax = float(selected["Coronene"]["z"].max())

    z_scale_names = ["C_Coronene", "CC_Coronene", "CCC_Coronene"]
    other_xmin = min(float(selected[name]["z"].min()) for name in z_scale_names)
    other_xmax = max(float(selected[name]["z"].max()) for name in z_scale_names)
    other_xmax = min(other_xmax, 4.0)

    ax_e = axes[0, 4]
    z_e = np.linspace(other_xmin, other_xmax, 90)
    fifth_col_y_values = []

    for name in ordered_names:
        _, E_pred_center = predict_line_for_xy(
            model_orb, name, z_e, xy=(0.0, 0.0), type_dict=type_dict, box=box
        )

        z_dense, E_dense = make_smooth_curve(z_e, E_pred_center, n=250)
        fifth_col_y_values.append(E_dense)
        ax_e.plot(
            z_dense,
            E_dense,
            linestyle="-",
            linewidth=1.0,
            color=colors[name],
            label=short_labels[name],
        )

    ax_e.tick_params(top=False, right=False)

    for col, name in enumerate(ordered_names):
        host_syms, host_pos = get_host(name)
        Z = np.array([type_dict[s] for s in host_syms] + [type_dict["He"]], dtype=int)

        xyz = selected[name]["xyz"]
        z = selected[name]["z"]
        E_ref = selected[name]["E_ref"]

        E_pred = predict_eV(model_ms, Z, host_pos, xyz, box) * CM1_PER_eV
        pred_ms_all[name] = E_pred

        plot_ref_pred(axes[1, col], z, E_ref, E_pred, colors[name], show_legend=True)

        n_atoms = len(host_syms) + 1
        mae = calc_mae(E_ref, E_pred)
        mae_per_atom = mae / n_atoms

        mae_table["MatterSim"][name] = (mae, mae_per_atom, n_atoms)

    ax_e2 = axes[1, 4]

    for name in ordered_names:
        _, E_pred_center = predict_line_for_xy(
            model_ms, name, z_e, xy=(0.0, 0.0), type_dict=type_dict, box=box
        )

        z_dense, E_dense = make_smooth_curve(z_e, E_pred_center, n=250)
        fifth_col_y_values.append(E_dense)
        ax_e2.plot(
            z_dense,
            E_dense,
            linestyle="-",
            linewidth=1.0,
            color=colors[name],
            label=short_labels[name],
        )

    ax_e2.tick_params(top=False, right=False)
    ax_e2.legend(
        frameon=False,
        loc="upper right",
        handlelength=0.8,
        labelspacing=0.2,
        columnspacing=0.8,
        ncol=2,
    )


    common_ylim = (-150.0, 149.0)

    axes[1, 0].set_ylim(common_ylim)
    axes[1, 0].yaxis.set_major_locator(MaxNLocator(5))
    fig.canvas.draw()

    common_ticks = axes[1, 0].get_yticks()

    if fifth_col_y_values:
        fifth_col_ymin = float(min(np.min(y) for y in fifth_col_y_values)) - 1.0
    else:
        fifth_col_ymin = common_ylim[0]
    fifth_col_ylim = (fifth_col_ymin, 220.0)

    cor_xticks = np.arange(np.ceil(cor_xmin), np.floor(cor_xmax) + 1, 1)
    other_xticks = np.arange(np.ceil(other_xmin), np.floor(other_xmax) + 1, 1)

    for row in range(2):
        for col in range(5):
            ax = axes[row, col]

            if col == 0:
                ax.set_xlim(cor_xmin, cor_xmax)
                ax.set_xticks(cor_xticks)
            else:
                ax.set_xlim(other_xmin, other_xmax)
                ax.set_xticks(other_xticks)

            if col == 4:
                ax.set_ylim(fifth_col_ylim)
                ax.yaxis.set_major_locator(MaxNLocator(5))
            else:
                ax.set_ylim(common_ylim)
                ax.set_yticks(common_ticks)

            ax.set_axisbelow(True)
            ax.grid(
                True,
                which="major",
                axis="both",
                linestyle="-",
                linewidth=0,
                alpha=0,
            )
            ax.tick_params(top=False, right=False)

    for row in range(2):
        for col in range(1, 4):
            axes[row, col].set_yticklabels([])

    for row in range(2):
        axes[row, 4].yaxis.set_ticks_position("right")
        axes[row, 4].tick_params(
            labelleft=False,
            labelright=True,
            left=False,
            right=True,
            top=False,
        )

    for row in range(2):
        axes[row, 0].set_ylabel(r"$E$ ($\mathrm{cm}^{-1}$)", labelpad=12)
        axes[row, 0].yaxis.set_label_coords(-0.28, 0.5)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    out_png = OUT_DIR / "combined_figure.png"
    out_svg = OUT_DIR / "combined_figure.svg"

    plt.savefig(out_png, dpi=600, bbox_inches="tight", transparent=False)
    plt.savefig(out_svg, dpi=600, bbox_inches="tight", transparent=False)

    plt.close(fig)

    print("\n" + "=" * 80)
    print("MAE summary (unit: cm^-1)")
    print("=" * 80)

    for model_name in ["Orb", "MatterSim"]:
        print(f"\n{model_name}")
        print("-" * 80)
        print(f"{'Structure':<15s} {'N_atoms':>8s} {'MAE':>15s} {'MAE/atom':>15s}")
        print("-" * 80)

        for name in ordered_names:
            mae, mae_per_atom, n_atoms = mae_table[model_name][name]
            print(f"{name:<15s} {n_atoms:8d} {mae:15.6f} {mae_per_atom:15.6f}")

    print("\nDone ->", out_png)
    print("Done ->", out_svg)


if __name__ == "__main__":
    main()