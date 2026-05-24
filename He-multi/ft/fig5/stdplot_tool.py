import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def get_cycle(cmap, N, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in [
                "Pastel1",
                "Pastel2",
                "Paired",
                "Accent",
                "Dark2",
                "Set1",
                "Set2",
                "Set3",
                "tab10",
                "tab20",
                "tab20b",
                "tab20c",
            ]:
                use_index = True
                cmap = mpl.colormaps[cmap]
                if not N:
                    N = cmap.N
            else:
                use_index = False
                cmap = mpl.colormaps[cmap]
                if not N:
                    N = 10

    if use_index == "auto":
        if cmap.N > 100:
            use_index = False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0, 1, N))
        return cycler("color", colors)


def mpl_std_Params(x, y=3/4, cmap=None, N=None):
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"] # "Arial"
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["figure.figsize"] = [x*8.27 ,x*y*8.27]  # centimeters in inches
    mpl.rcParams["lines.linewidth"] = 1
    mpl.rcParams["lines.marker"] = "None"
    mpl.rcParams["lines.markersize"] = 0.5
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{bm}\usepackage{newtxtext,newtxmath}"

    if cmap:
        if type(cmap) == str:
            mpl.rcParams["axes.prop_cycle"] = get_cycle(cmap, N)
        elif type(cmap) == list:
            mpl.rcParams["axes.prop_cycle"] = cycler("color", cmap)
