import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq

try:
    from scipy.special import sph_harm_y as sph_harm
except Exception:
    from scipy.special import sph_harm as sph_harm


SAPT_PATH = "sapt_2200.npy"
OUT_NPZ = "Rcut_full_disp.npz"
OUT_PNG = "Rcut_full_disp_heatmap.png"

N_THETA = 80
N_PHI_POS = 120
N_PHI_FULL = 721
N_SCAN_R = 80

C6_PERIOD = 2.0 * np.pi / 6.0
HALF_PERIOD = C6_PERIOD / 2.0


def parse_xyz(dist_str):
    x, y, z = dist_str.split("___")
    return float(x), float(y), float(z)


sapt_res = np.load(SAPT_PATH, allow_pickle=True).item()
basis = "QZ" if "QZ" in sapt_res else list(sapt_res.keys())[0]
entries = sapt_res[basis]

X = np.array([parse_xyz(e["Distance"]) for e in entries], dtype=float)

E_elst = np.array([e["Electrostatics"] for e in entries], dtype=float)
E_exch = np.array([e["Exchange"] for e in entries], dtype=float)
E_ind = np.array([e["Induction"] for e in entries], dtype=float)
E_disp = np.array([e["Dispersion"] for e in entries], dtype=float)

x, y, z = X[:, 0], X[:, 1], X[:, 2]
R_data = np.sqrt(x * x + y * y + z * z)

phi_data = np.mod(np.arctan2(y, x), 2.0 * np.pi)
theta_data = np.arccos(np.clip(z / np.maximum(R_data, 1e-12), -1.0, 1.0))

theta_min = float(theta_data.min())
theta_max = float(theta_data.max())

R_lo_global = float(np.quantile(R_data, 0.02))
R_hi_global = float(np.quantile(R_data, 0.98))

Y = np.column_stack([E_elst, E_exch, E_ind, E_disp])
neighbors = min(200, len(X))

rbf = RBFInterpolator(
    X,
    Y,
    neighbors=neighbors,
    kernel="thin_plate_spline",
    smoothing=1e-6,
)


def g_batch_R(theta, phi, Rs):
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    Rs = np.asarray(Rs, dtype=float)
    cph = np.cos(phi)
    sph = np.sin(phi)

    pts = np.empty((Rs.size, 3), dtype=float)
    pts[:, 0] = Rs * sin_th * cph
    pts[:, 1] = Rs * sin_th * sph
    pts[:, 2] = Rs * cos_th

    out = rbf(pts)

    elst = out[:, 0]
    exch = out[:, 1]
    ind = out[:, 2]
    disp = out[:, 3]

    return np.abs(elst + exch + ind) - np.abs(disp)


def g_single_R(theta, phi, R):
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    pt = np.array(
        [[
            R * sin_th * np.cos(phi),
            R * sin_th * np.sin(phi),
            R * cos_th,
        ]],
        dtype=float,
    )

    elst, exch, ind, disp = rbf(pt)[0]
    return abs(elst + exch + ind) - abs(disp)


def find_root_for_theta_phi(theta, phi, R_lo, R_hi, n_scan=N_SCAN_R):
    Rs = np.linspace(R_lo, R_hi, n_scan)
    gs = g_batch_R(theta, phi, Rs)
    sgn = np.sign(gs)

    for i in range(len(Rs) - 1):
        if np.isfinite(gs[i]) and np.isfinite(gs[i + 1]) and sgn[i] * sgn[i + 1] < 0:
            return brentq(
                lambda RR: g_single_R(theta, phi, RR),
                Rs[i],
                Rs[i + 1],
                maxiter=80,
            )

    return np.nan


def phi_to_base_abs(phi):
    return abs(((phi + HALF_PERIOD) % C6_PERIOD) - HALF_PERIOD)


phi_pos_max_from_data = float(np.quantile(phi_data, 0.98))
phi_pos_max = min(max(phi_pos_max_from_data, 0.0), HALF_PERIOD)

thetas = np.linspace(theta_min, theta_max, N_THETA)
phis_pos = np.linspace(0.0, phi_pos_max, N_PHI_POS)
phis_full = np.linspace(0.0, 2.0 * np.pi, N_PHI_FULL, endpoint=False)

phi_base_abs = np.array([phi_to_base_abs(ph) for ph in phis_full], dtype=float)

R_cut_pos = np.full((N_THETA, N_PHI_POS), np.nan, dtype=float)

for it, theta in enumerate(thetas):
    for ip, phi in enumerate(phis_pos):
        R_cut_pos[it, ip] = find_root_for_theta_phi(
            theta,
            phi,
            R_lo_global,
            R_hi_global,
            n_scan=N_SCAN_R,
        )

R_cut_full = np.full((N_THETA, N_PHI_FULL), np.nan, dtype=float)

for it in range(N_THETA):
    yv = R_cut_pos[it]
    mask = np.isfinite(yv)

    if mask.sum() >= 2:
        R_cut_full[it] = np.interp(
            phi_base_abs,
            phis_pos[mask],
            yv[mask],
            left=np.nan,
            right=np.nan,
        )

dtheta = thetas[1] - thetas[0] if N_THETA > 1 else 1.0

for it, theta in enumerate(thetas):
    theta_mirror = np.pi - theta

    if theta_mirror < thetas[0] or theta_mirror > thetas[-1]:
        continue

    jt = int(np.round((theta_mirror - thetas[0]) / dtheta))

    if jt < 0 or jt >= N_THETA or jt < it:
        continue

    a = R_cut_full[it]
    b = R_cut_full[jt]
    both = np.isfinite(a) & np.isfinite(b)

    if np.any(both):
        avg = 0.5 * (a[both] + b[both])
        a_new = a.copy()
        b_new = b.copy()
        a_new[both] = avg
        b_new[both] = avg
        R_cut_full[it] = a_new
        R_cut_full[jt] = b_new

np.savez(
    OUT_NPZ,
    thetas=thetas,
    phis=phis_full,
    R_cut_map=R_cut_full,
    basis=basis,
)

PHI, THETA = np.meshgrid(phis_full, thetas)

plt.figure(figsize=(10, 5))
pm = plt.pcolormesh(PHI, THETA, R_cut_full, shading="auto")
plt.colorbar(pm, label=r"$R_{\mathrm{cut}}$ ($\mathrm{\AA}$)")
plt.xlabel(r"$\phi$ (rad)")
plt.ylabel(r"$\theta$ (rad)")
plt.title(r"$R_{\mathrm{cut}}(\theta,\phi)$")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()