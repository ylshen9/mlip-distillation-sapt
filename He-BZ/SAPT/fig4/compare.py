# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from stdplot_tool import mpl_std_Params

import re
import math
import colorsys
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import colors as mcolors

try:
    from scipy.special import sph_harm_y as sph_harm
except Exception:
    from scipy.special import sph_harm as sph_harm

from train import BindingEnergyTransformer


RC_NPZ_PATH = "./rc_fit.npz"

REPEAT_DIRS = ["check", "check1", "check2", "check3", "check4", "check5","check6", "check7", "check8", "check9", "check10", "check11"]

DIRECT_ROOT = Path("direct")
ORB_ROOT = Path("ft")
DFT_ROOT = Path("dft")

VALIDATION_DIR = Path("validation")

OUT_PNG = "compare.png"
OUT_PDF = "compare.pdf"
OUT_TXT = "compare.txt"

CELL = [20.0, 20.0, 20.0]
FAR_Z = 12.0
CM1_PER_eV = 8065.544

D_MODEL = 576
NUM_LAYERS = 8
SOG_N_GAUSSIANS = 12
N_RBF = 28
RC_MIN = 0.8
RC_MAX = 12.0


BZ_SYMS = ["C"] * 6 + ["H"] * 6
BZ_POS = np.array(
    [
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
    ],
    dtype=np.float32,
)

HE_ATOM_INDEX = len(BZ_SYMS)

type_dict = {"C": 0, "H": 1, "He": 2}
type_list = ["C", "H", "He"]

Z_np = np.array([type_dict[s] for s in BZ_SYMS] + [type_dict["He"]], dtype=np.int64)
Z_t = torch.tensor(Z_np, dtype=torch.long).unsqueeze(0)

box_mat = np.array(
    [
        [CELL[0], 0.0, 0.0],
        [0.0, CELL[1], 0.0],
        [0.0, 0.0, CELL[2]],
    ],
    dtype=np.float32,
)

bz_center = BZ_POS.mean(axis=0)


def lighten_color(color, amount=0.55):

    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - amount * (1 - l)
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return (rr, gg, bb)


def _sph_eval(m, l, phi, theta):
    if getattr(sph_harm, "__name__", "") == "sph_harm_y":
        return sph_harm(m, l, theta, phi)
    return sph_harm(m, l, phi, theta)


def _sph_real_design(theta, phi, lmax: int):
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


def load_rc_model(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    lmax = int(np.asarray(d["lmax"]).item())
    coeff = np.asarray(d["coeff"], float)
    return {"lmax": lmax, "coeff": coeff}


rc_model = load_rc_model(RC_NPZ_PATH)


def rc_for_he(he_pos):
    v = np.asarray(he_pos, dtype=np.float32) - bz_center
    r = float(np.linalg.norm(v))
    r = max(r, 1e-12)
    theta = float(np.arccos(np.clip(v[2] / r, -1.0, 1.0)))
    phi = float(np.mod(np.arctan2(v[1], v[0]), 2.0 * np.pi))
    A = _sph_real_design(np.array([theta]), np.array([phi]), rc_model["lmax"])
    rc = float((A @ rc_model["coeff"]).reshape(-1)[0])
    return max(rc, 1e-3)


def build_model():
    return BindingEnergyTransformer(
        n_types=len(type_list),
        he_index=HE_ATOM_INDEX,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        sog_n_gaussians=SOG_N_GAUSSIANS,
        n_rbf=N_RBF,
        rc_min=RC_MIN,
        rc_max=RC_MAX,
        rc_lr=12,
    )


def get_validation_file_for_repeat(repeat_name: str) -> Path:
    val_file = VALIDATION_DIR / f"{repeat_name}.npy"
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    return val_file


def load_validation_xyz_E(validation_npy: Path):
    arr = np.load(validation_npy)
    xyz_val = arr[:, :3].astype(np.float32)
    Eb_ref_val_cm1 = arr[:, 3].astype(np.float64)
    return xyz_val, Eb_ref_val_cm1


def eval_one_checkpoint(model_pt: Path, xyz_val: np.ndarray, Eb_ref_val: np.ndarray):
    model = build_model()
    state = torch.load(str(model_pt), map_location="cpu")
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

    _ = model_energy_eV([0.0, 0.0, FAR_Z])

    Eb_pred_val = np.empty(len(xyz_val), dtype=np.float64)
    for i, (x, y, z) in enumerate(xyz_val):
        E_tot = model_energy_eV([float(x), float(y), float(z)])
        Eb_pred_val[i] = E_tot * CM1_PER_eV

    mae_val = float(np.mean(np.abs(Eb_pred_val - Eb_ref_val)))
    return mae_val


def parse_percent_from_name(p: Path) -> int:
    m = re.match(r"(\d+)\.pt$", p.name)
    if m is None:
        raise ValueError(f"Invalid checkpoint name: {p.name}")
    return int(m.group(1))


def collect_one_repeat_series(model_dir: Path, label: str, xyz_val: np.ndarray, Eb_ref_val: np.ndarray):

    result = {}
    pts = [p for p in model_dir.glob("*.pt") if p.is_file()]
    pts = [p for p in pts if parse_percent_from_name(p) in range(10, 81, 10)]
    pts = sorted(pts, key=parse_percent_from_name)

    for pt in pts:
        percent = parse_percent_from_name(pt)
        mae_val = eval_one_checkpoint(pt, xyz_val, Eb_ref_val)
        result[percent] = mae_val

        print(
            f"{label:12s}  {str(model_dir):20s}  "
            f"val={len(xyz_val):>5d}  {percent:>3d}%  "
            f"MAE(val)={mae_val:10.4f}"
        )

    return result


def collect_series_with_repeats(root_dir: Path, label: str):

    per_repeat = []

    for sub in REPEAT_DIRS:
        run_dir = root_dir / sub
        if not run_dir.exists():
            print(f"[WARN] Missing repeat dir: {run_dir}")
            continue

        val_file = get_validation_file_for_repeat(sub)
        xyz_val, Eb_ref_val = load_validation_xyz_E(val_file)

        run_result = collect_one_repeat_series(run_dir, label, xyz_val, Eb_ref_val)
        if len(run_result) > 0:
            per_repeat.append({
                "repeat": sub,
                "validation_file": str(val_file),
                "values": run_result,
            })

    if len(per_repeat) == 0:
        raise RuntimeError(f"No valid checkpoints found under {root_dir}")

    all_percents = sorted(
        set().union(*[set(d["values"].keys()) for d in per_repeat])
    )

    xs = []
    y_mean = []
    y_std = []
    y_sem = []
    y_n = []

    for p in all_percents:
        vals = [d["values"][p] for d in per_repeat if p in d["values"]]
        if len(vals) == 0:
            continue

        vals = np.array(vals, dtype=float)
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0

        xs.append(p)
        y_mean.append(mean)
        y_std.append(std)
        y_sem.append(sem)
        y_n.append(n)

    return {
        "x": np.array(xs, dtype=int),
        "mean": np.array(y_mean, dtype=float),
        "std": np.array(y_std, dtype=float),
        "sem": np.array(y_sem, dtype=float),
        "n": np.array(y_n, dtype=int),
        "raw": per_repeat,
    }


def save_compare_txt(series_dict, out_txt):

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Summary\n")
        f.write("# percent  method  mean_MAE_cm^-1  std_MAE_cm^-1  sem_MAE_cm^-1  n_runs\n")
        for method, d in series_dict.items():
            for x, m, s, se, n in zip(d["x"], d["mean"], d["std"], d["sem"], d["n"]):
                f.write(
                    f"{x:3d}  {method:14s}  "
                    f"{m:14.8f}  {s:14.8f}  {se:14.8f}  {n:d}\n"
                )

        f.write("\n# Raw per-repeat values\n")
        f.write("# method  repeat  validation_file  percent  mae_cm^-1\n")
        for method, d in series_dict.items():
            for item in d["raw"]:
                repeat_name = item["repeat"]
                val_file = item["validation_file"]
                for percent in sorted(item["values"].keys()):
                    mae = item["values"][percent]
                    f.write(
                        f"{method:14s}  {repeat_name:8s}  {val_file:24s}  "
                        f"{percent:3d}  {mae:14.8f}\n"
                    )


def main():
    series = {
        "CCSD": collect_series_with_repeats(DIRECT_ROOT, "CCSD"),
        "DFT+CCSD": collect_series_with_repeats(DFT_ROOT, "DFT+CCSD"),
        "UMLFF+CCSD": collect_series_with_repeats(ORB_ROOT, "UMLFF+CCSD"),
    }

    save_compare_txt(series, OUT_TXT)

    mpl_std_Params(x=0.5, cmap="Set2")
    fig, ax = plt.subplots()

    style_map = {
        "CCSD": {"color": "C0", "marker": "o"},
        "DFT+CCSD": {"color": "C1", "marker": "^"},
        "UMLFF+CCSD": {"color": "C2", "marker": "s"},
    }

    all_means = []
    all_sems = []
    all_xs = []

    for method, d in series.items():
        all_means.extend(d["mean"].tolist())
        all_sems.extend(d["sem"].tolist())
        all_xs.extend(d["x"].tolist())

    ymin = min(np.array(all_means) - np.array(all_sems))
    ymax = max(np.array(all_means) + np.array(all_sems))

    y_bottom = 4 * math.floor(ymin / 4.0)
    y_top = 4 * math.ceil(ymax / 4.0)
    if y_bottom == y_top:
        y_top = y_bottom + 2

    offset_map = {
        "CCSD": -0.7,
        "DFT+CCSD": 0.0,
        "UMLFF+CCSD": 0.7,
    }

    for method, d in series.items():
        color = style_map[method]["color"]
        marker = style_map[method]["marker"]
        marker_face = lighten_color(color, amount=0.68)
        x_plot = d["x"] + offset_map[method]

        ax.plot(
            x_plot,
            d["mean"],
            color=color,
            linewidth=1.8,
            zorder=1,
        )

        ax.errorbar(
            x_plot,
            d["mean"],
            yerr=d["sem"],
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=0,
            zorder=2,
        )

        ax.scatter(
            x_plot,
            d["mean"],
            s=24,
            marker=marker,
            facecolors=marker_face,
            edgecolors=color,
            linewidths=0.9,
            zorder=3,
            label=method,
        )

    ax.set_xlabel("Percent of data (\\%)")
    ax.set_ylabel(r"MAE (cm$^{-1}$)")

    ax.set_xticks(sorted(set(all_xs)))
    ax.set_ylim(y_bottom, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(4))

    ax.tick_params(top=False, right=False)
    ax.grid(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=True)
    plt.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    plt.savefig(Path(OUT_PNG).with_suffix(".svg"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(OUT_TXT)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()