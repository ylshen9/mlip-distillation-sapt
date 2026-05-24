import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stdplot_tool import mpl_std_Params


SAPT_PATH = "sapt_2200.npy"
OUT_PNG = "sapt_components_z_axis.png"

TOL = 1e-6
X_MAX = 3.0


def parse_xyz(dist_str):
    x, y, z = dist_str.split("___")
    return float(x), float(y), float(z)


sapt_res = np.load(SAPT_PATH, allow_pickle=True).item()
basis = "QZ" if "QZ" in sapt_res else list(sapt_res.keys())[0]
entries = sapt_res[basis]

X = np.array([parse_xyz(e["Distance"]) for e in entries], dtype=float)

x = X[:, 0]
y = X[:, 1]
z = X[:, 2]

E_elst = np.array([e["Electrostatics"] for e in entries], dtype=float)
E_exch = np.array([e["Exchange"] for e in entries], dtype=float)
E_ind = np.array([e["Induction"] for e in entries], dtype=float)
E_disp = np.array([e["Dispersion"] for e in entries], dtype=float)

mask = (np.abs(x) <= TOL) & (np.abs(y) <= TOL)

z_sel = z[mask]
elst_sel = E_elst[mask]
exch_sel = E_exch[mask]
ind_sel = E_ind[mask]
disp_sel = E_disp[mask]

idx = np.argsort(z_sel)

z_sel = z_sel[idx]
elst_sel = elst_sel[idx]
exch_sel = exch_sel[idx]
ind_sel = ind_sel[idx]
disp_sel = disp_sel[idx]

lhs = np.abs(elst_sel + exch_sel + ind_sel)
rhs = np.abs(disp_sel)
diff = lhs - rhs

z_cutoff = None

for i in range(len(diff) - 1):
    if diff[i] * diff[i + 1] <= 0:
        dz = z_sel[i + 1] - z_sel[i]
        dd = diff[i + 1] - diff[i]
        z_cutoff = z_sel[i] - diff[i] * dz / dd
        break

mpl_std_Params(0.45, y=1, cmap="Set2")

fig, ax = plt.subplots()

ax.plot(z_sel, elst_sel, marker=".", markersize=3, linewidth=1.0, label="Electrostatics")
ax.plot(z_sel, exch_sel, marker=".", markersize=3, linewidth=1.0, label="Exchange")
ax.plot(z_sel, ind_sel, marker=".", markersize=3, linewidth=1.0, label="Induction")
ax.plot(z_sel, disp_sel, marker=".", markersize=3, linewidth=1.0, label="Dispersion")

ax.axhline(0.0, linewidth=0.8, color="black")

if z_cutoff is not None:
    ax.axvline(
        z_cutoff,
        linestyle="--",
        linewidth=0.8,
        color="black",
        zorder=1,
        label=f"$r_{{\\mathrm{{c}}}}$ = {z_cutoff:.2f} Å",
    )

ax.set_xlim(z_sel.min(), X_MAX)
ax.set_xlabel(r"$z$ (Å)")
ax.set_ylabel(r"Energy (kcal mol$^{-1}$)")
ax.tick_params(top=False, right=False)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=True)
plt.savefig(Path(OUT_PNG).with_suffix(".svg"), dpi=600, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"Saved: {OUT_PNG}")