import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import colorsys

from stdplot_tool import mpl_std_Params


TXT_FILE = "compare.txt"
OUT_PNG = "compare.png"
OUT_PDF = "compare.pdf"


def lighten_color(color, amount=0.55):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - amount * (1 - l)
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return (rr, gg, bb)


def read_compare_txt(txt_file):
    summary = {}

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) != 6:
                continue

            try:
                percent = int(parts[0])
                method = parts[1]
                mean = float(parts[2])
                std = float(parts[3])
                sem = float(parts[4])
                n = int(parts[5])
            except ValueError:
                continue

            if method not in summary:
                summary[method] = {"x": [], "mean": [], "std": [], "sem": [], "n": []}

            summary[method]["x"].append(percent)
            summary[method]["mean"].append(mean)
            summary[method]["std"].append(std)
            summary[method]["sem"].append(sem)
            summary[method]["n"].append(n)

    for method in summary:
        order = np.argsort(summary[method]["x"])
        for key in ["x", "mean", "std", "sem", "n"]:
            arr = np.array(summary[method][key])
            summary[method][key] = arr[order]

    return summary


def draw_series(ax, method, d, style_map, zorder_map, x_offset=0.0):
    color = style_map[method]["color"]
    marker = style_map[method]["marker"]
    marker_face = lighten_color(color, amount=0.55)
    x_plot = d["x"].astype(float) + x_offset

    ax.plot(
        x_plot, d["mean"],
        color=color, linewidth=1.5,
        zorder=zorder_map[method]["line"],
    )
    ax.scatter(
        x_plot, d["mean"],
        s=14, marker=marker,
        facecolors=marker_face, edgecolors=color,
        linewidths=1.0,
        zorder=zorder_map[method]["scatter"],
        label=method,
    )
    ax.errorbar(
        x_plot, d["mean"], yerr=d["std"],
        fmt="none", ecolor=color,
        elinewidth=0.8, capsize=0,
        barsabove=True,
        zorder=zorder_map[method]["errorbar"],
    )


def main():
    series = read_compare_txt(TXT_FILE)

    mpl_std_Params(x=0.5, cmap="Set2")
    fig, ax = plt.subplots()

    plot_order = ["CCSD", "DFT+CCSD", "UMLFF+CCSD"]


    style_map = {
        "CCSD": {"color": "C0", "marker": "o"},
        "DFT+CCSD": {"color": "C1", "marker": "^"},
        "UMLFF+CCSD": {"color": "C2", "marker": "s"},
    }

    zorder_map = {
        "CCSD": {"line": 1, "scatter": 4, "errorbar": 7},
        "DFT+CCSD": {"line": 2, "scatter": 5, "errorbar": 8},
        "UMLFF+CCSD": {"line": 3, "scatter": 6, "errorbar": 9},
    }


    legend_label_map = {
        "CCSD": "CCSD(T)",
        "DFT+CCSD": "DFT+CCSD(T)",
        "UMLFF+CCSD": "MLIP+CCSD(T)",
    }

    all_xs = []
    ymins_all = []
    ymaxs_all = []

    for method in plot_order:
        if method not in series:
            continue
        d = series[method]
        all_xs.extend(d["x"].tolist())
        ymins_all.extend((d["mean"] - d["std"]).tolist())
        ymaxs_all.extend((d["mean"] + d["std"]).tolist())

    if not all_xs:
        raise ValueError("No valid data found in compare.txt")

    ymin_data = min(ymins_all)
    ymax_data = max(ymaxs_all)


    y_bottom = 4 * math.floor(ymin_data / 4.0)
    y_top = ymax_data + 0.1


    for method in plot_order:
        if method not in series:
            continue
        d = series[method]
        color = style_map[method]["color"]
        ax.plot(d["x"].astype(float), d["mean"],
                color=color, linewidth=1.5,
                zorder=zorder_map[method]["line"])


    for method in plot_order:
        if method not in series:
            continue
        d = series[method]
        color = style_map[method]["color"]
        marker_face = lighten_color(color, amount=0.55)
        ax.scatter(d["x"].astype(float), d["mean"],
                   s=14, marker=style_map[method]["marker"],
                   facecolors=marker_face, edgecolors=color,
                   linewidths=1.0, label=legend_label_map[method],
                   zorder=zorder_map[method]["scatter"])


    for method in plot_order:
        if method not in series:
            continue
        d = series[method]
        ax.errorbar(d["x"].astype(float), d["mean"], yerr=d["std"],
                    fmt="none", ecolor=style_map[method]["color"],
                    elinewidth=0.8, capsize=0, barsabove=True,
                    zorder=zorder_map[method]["errorbar"])

    ax.set_xlabel("Percent of data (\\%)")
    ax.set_ylabel(r"MAE (cm$^{-1}$)")
    ax.set_xlim(left=None, right=81)
    ax.set_xticks(sorted(set(all_xs)))
    ax.set_ylim(y_bottom, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.tick_params(top=False, right=False)
    ax.grid(False)


    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    ordered_labels = [
        legend_label_map[m] for m in plot_order
        if legend_label_map[m] in label_to_handle
    ]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]

    leg = ax.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        loc="upper right"
    )


    ax_inset = ax.inset_axes(
        [0.50, 0.42, 0.44, 0.22],
        transform=ax.transAxes,
    )

    zoom_methods = ["DFT+CCSD", "UMLFF+CCSD"]
    zoom_xs_all = []
    zoom_ymins = []
    zoom_ymaxs = []

    for method in zoom_methods:
        if method not in series:
            continue
        d = series[method]
        mask = d["x"] >= 50
        if not np.any(mask):
            continue

        xz = d["x"][mask].astype(float)
        mz = d["mean"][mask]
        sz = d["sem"][mask]
        color = style_map[method]["color"]
        marker_face = lighten_color(color, amount=0.55)

        ax_inset.plot(xz, mz, color=color, linewidth=1.2,
                      zorder=zorder_map[method]["line"])
        ax_inset.scatter(xz, mz, s=10,
                         marker=style_map[method]["marker"],
                         facecolors=marker_face, edgecolors=color,
                         linewidths=0.8,
                         zorder=zorder_map[method]["scatter"])
        ax_inset.errorbar(xz, mz, yerr=sz,
                          fmt="none", ecolor=color,
                          elinewidth=0.7, capsize=0,
                          zorder=zorder_map[method]["errorbar"])

        zoom_xs_all.extend(xz.tolist())
        zoom_ymins.extend((mz - sz).tolist())
        zoom_ymaxs.extend((mz + sz).tolist())

    if zoom_xs_all:
        zy_min = min(zoom_ymins)
        zy_max = max(zoom_ymaxs)
        zy_pad = (zy_max - zy_min) * 0.15 if (zy_max - zy_min) > 0 else 0.2
        ax_inset.set_xlim(48, 82)
        ax_inset.set_ylim(zy_min - zy_pad, zy_max + zy_pad + 0.1)
        ax_inset.set_xticks([50, 60, 70, 80])


        ax_inset.tick_params(labelsize=6, top=False, right=False,
                             length=2, width=0.6)
        for spine in ax_inset.spines.values():
            spine.set_linewidth(0.6)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax_inset.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=True)
    plt.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    plt.savefig(Path(OUT_PNG).with_suffix(".svg"),
                dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()