import numpy as np
from scipy.special import sph_harm


INPUT_NPZ = "Rcut_full_disp.npz"
OUT_NPZ = "rc_fit.npz"
LMAX = 10


data = np.load(INPUT_NPZ, allow_pickle=True)

thetas = data["thetas"].astype(float)
phis = data["phis"].astype(float)
Rmap = data["R_cut_map"].astype(float)

PHI, THETA = np.meshgrid(phis, thetas)
mask = np.isfinite(Rmap)

theta0 = THETA[mask]
phi0 = PHI[mask]
R0 = Rmap[mask]

theta_v = np.concatenate([theta0, np.pi - theta0])
phi_v = np.concatenate([phi0, phi0])
R_v = np.concatenate([R0, R0])


def design_real_sph(theta, phi, lmax):
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    cols = []
    meta = []

    for l in range(lmax + 1):
        Y0 = sph_harm(0, l, phi, theta)
        cols.append(np.real(Y0))
        meta.append((l, 0, "m0"))

        for m in range(1, l + 1):
            Y = sph_harm(m, l, phi, theta)

            cols.append(np.sqrt(2.0) * ((-1) ** m) * np.real(Y))
            meta.append((l, m, "c"))

            cols.append(np.sqrt(2.0) * ((-1) ** m) * np.imag(Y))
            meta.append((l, m, "s"))

    return np.column_stack(cols), np.array(meta, dtype=object)


A, meta = design_real_sph(theta_v, phi_v, LMAX)

coeff, *_ = np.linalg.lstsq(A, R_v, rcond=None)

R_pred = A @ coeff
mae = float(np.mean(np.abs(R_pred - R_v)))

np.savez(
    OUT_NPZ,
    lmax=int(LMAX),
    coeff=coeff.astype(float),
    meta=meta,
    mae_existing=mae,
)

print(f"MAE = {mae:.6f} Å")
print(f"Saved: {OUT_NPZ}")