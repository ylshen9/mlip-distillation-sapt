import math
import numpy as np
from pathlib import Path

OUT_ROOT = "dataset"
CELL_A = 40.0
SEED = 42
TRAIN_RATIO = 0.8
E_ABS_MAX_CM1 = 1000.0
MULTI_NPY = "../multi.npy"
MULTI_METHOD = "pbe0_def2-SVP"
CORONENE_NPY = "../coronene.npy"
CORONENE_N_USE = None
TYPE_MAP = ["C", "H", "He"]
type_idx = {"C": 0, "H": 1, "He": 2}
CM1_PER_EV = 8065.544005
MAX_FRAC = 0.5

BZ_POS = np.array([
    [-1.2073830, -0.6970829, 0.0], [-1.2073830,  0.6970829, 0.0], [0.0,  1.3941659, 0.0],
    [ 1.2073830,  0.6970829, 0.0], [ 1.2073830, -0.6970829, 0.0], [0.0, -1.3941659, 0.0],
    [-2.1490090, -1.2407309, 0.0], [-2.1490090,  1.2407309, 0.0], [0.0,  2.4814619, 0.0],
    [ 2.1490090,  1.2407309, 0.0], [ 2.1490090, -1.2407309, 0.0], [0.0, -2.4814619, 0.0],
], dtype=float)

BZ_C = BZ_POS[:6, :2]
BZ_H = BZ_POS[6:, :2]
R_C = np.linalg.norm(BZ_C, axis=1).mean()
R_H = np.linalg.norm(BZ_H, axis=1).mean()
R_CH = R_H - R_C
L = R_C * math.sqrt(3.0)
a1 = np.array([L, 0.0], dtype=float)
a2 = np.array([L * 0.5, L * math.sqrt(3) / 2.0], dtype=float)

CORONENE_SYMS = ["C"] * 24 + ["H"] * 12
CORONENE_POS = np.array([
    [1.211568, 3.492552, 0.0],
    [2.418855, 2.795525, 0.0],
    [2.424434, 1.399748, 0.0],
    [3.630423, 0.697027, 0.0],
    [3.630423, -0.697027, 0.0],
    [2.424434, -1.399748, 0.0],
    [2.418855, -2.795525, 0.0],
    [1.211568, -3.492552, 0.0],
    [-0.000000, -2.799496, 0.0],
    [-1.211568, -3.492552, 0.0],
    [-2.418855, -2.795525, 0.0],
    [-2.424434, -1.399748, 0.0],
    [-3.630423, -0.697027, 0.0],
    [-3.630423, 0.697027, 0.0],
    [-2.424434, 1.399748, 0.0],
    [-2.418855, 2.795525, 0.0],
    [-1.211568, 3.492552, 0.0],
    [-0.000000, 2.799496, 0.0],
    [-0.000000, 1.399577, 0.0],
    [-1.212069, 0.699788, 0.0],
    [-1.212069, -0.699788, 0.0],
    [-0.000000, -1.399577, 0.0],
    [1.212069, -0.699788, 0.0],
    [1.212069, 0.699788, 0.0],
    [1.220986, 4.580106, 0.0],
    [3.355996, 3.347458, 0.0],
    [4.576981, 1.232649, 0.0],
    [4.576981, -1.232649, 0.0],
    [3.355996, -3.347458, 0.0],
    [1.220986, -4.580106, 0.0],
    [-1.220986, -4.580106, 0.0],
    [-3.355996, -3.347458, 0.0],
    [-4.576981, -1.232649, 0.0],
    [-4.576981, 1.232649, 0.0],
    [-3.355996, 3.347458, 0.0],
    [-1.220986, 4.580106, 0.0],
], dtype=float)

def make_hex(n_ring, shift=(0.0, 0.0, 0.0)):
    shift = np.asarray(shift, dtype=float)
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
            ang = math.pi / 2 + k * math.pi / 3
            C_list.append([cx + R_C * math.cos(ang), cy + R_C * math.sin(ang)])

    table = {}
    for x, y in C_list:
        key = (round(x / 1e-4) * 1e-4, round(y / 1e-4) * 1e-4)
        table.setdefault(key, [x, y])

    C_2d = np.array(list(table.values()), dtype=float)
    N_C = len(C_2d)

    neighbors = [[] for _ in range(N_C)]
    for i in range(N_C):
        for j in range(i + 1, N_C):
            d = np.linalg.norm(C_2d[i] - C_2d[j])
            if 0.8 * R_C < d < 1.2 * R_C:
                neighbors[i].append(j)
                neighbors[j].append(i)

    H_list = []
    for i in range(N_C):
        if len(neighbors[i]) < 3:
            v_sum = np.zeros(2, dtype=float)
            for j in neighbors[i]:
                v_sum += C_2d[j] - C_2d[i]
            out = -v_sum / np.linalg.norm(v_sum)
            H_list.append(C_2d[i] + out * R_CH)

    H_2d = np.array(H_list, dtype=float)
    C_3d = np.column_stack([C_2d, np.zeros(len(C_2d))]) + shift
    H_3d = np.column_stack([H_2d, np.zeros(len(H_2d))]) + shift

    syms = ["C"] * len(C_3d) + ["H"] * len(H_3d)
    pos = np.vstack([C_3d, H_3d]).astype(float)
    return syms, pos

def as_xyze(arr):
    return np.asarray(arr)[:, :4].astype(float)

def filter_by_energy_cm1(arr4):
    return arr4[np.abs(arr4[:, 3]) < E_ABS_MAX_CM1]

def remove_forbidden_points(arr4, tol=0.1):
    x = arr4[:, 0]
    y = arr4[:, 1]
    near_x2_y0 = (np.abs(x - 2.0) <= tol) & (np.abs(y) <= tol)
    near_x0_y0 = (np.abs(x) <= tol) & (np.abs(y) <= tol)
    return arr4[~(near_x2_y0 | near_x0_y0)]

def make_groups(xyz, n_r=8, n_theta=6, z_tol=1e-8):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    z_key = np.round(z / z_tol).astype(int)
    z_unique = np.unique(z_key)
    z_map = {v: i for i, v in enumerate(z_unique)}
    z_bin = np.array([z_map[v] for v in z_key], dtype=int)

    r_edges = np.linspace(r.min(), r.max(), n_r + 1)
    theta_edges = np.linspace(theta.min(), theta.max(), n_theta + 1)

    r_bin = np.clip(np.digitize(r, r_edges[1:-1]), 0, n_r - 1)
    theta_bin = np.clip(np.digitize(theta, theta_edges[1:-1]), 0, n_theta - 1)

    return (z_bin * (n_r * n_theta) + r_bin * n_theta + theta_bin).astype(int)

def split_by_cross_block(arr4, seed, max_frac=0.5, train_ratio=0.8):
    xyz = arr4[:, :3]
    groups = make_groups(xyz)

    rng = np.random.default_rng(seed)
    unique_groups = rng.permutation(np.unique(groups))

    n_pool_groups = int(len(unique_groups) * max_frac)
    n_pool_groups = max(1, min(n_pool_groups, len(unique_groups) - 1))

    pool_groups = unique_groups[:n_pool_groups]
    independent_val_groups = unique_groups[n_pool_groups:]

    pool_mask = np.isin(groups, pool_groups)
    independent_val_mask = np.isin(groups, independent_val_groups)

    arr4_pool = arr4[pool_mask]
    groups_pool = groups[pool_mask]
    arr4_independent_val = arr4[independent_val_mask]

    pool_groups_order = []
    seen = set()
    for g in pool_groups:
        if g not in seen:
            seen.add(g)
            pool_groups_order.append(g)
    pool_groups_order = np.array(pool_groups_order, dtype=int)

    n_train_groups = int(len(pool_groups_order) * train_ratio)
    n_train_groups = max(1, min(n_train_groups, len(pool_groups_order) - 1))

    train_groups = pool_groups_order[:n_train_groups]
    val_groups = pool_groups_order[n_train_groups:]

    train_mask = np.isin(groups_pool, train_groups)
    val_mask = np.isin(groups_pool, val_groups)

    train4 = arr4_pool[train_mask]
    val4 = arr4_pool[val_mask]

    return train4, val4, arr4_independent_val, train_groups, val_groups, independent_val_groups

def write_system(system_dir, set_name, host_syms, host_pos, xyz_subset, E_subset_ev):
    set_dir = system_dir / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    syms = list(host_syms) + ["He"]
    nat = len(syms)
    nframes = xyz_subset.shape[0]

    coords = np.zeros((nframes, nat, 3), dtype=np.float32)
    boxes = np.zeros((nframes, 3, 3), dtype=np.float32)
    energies = np.array(E_subset_ev, dtype=np.float64)
    box_single = np.diag([CELL_A, CELL_A, CELL_A]).astype(np.float32)
    host_pos = np.asarray(host_pos, dtype=np.float32)

    for i in range(nframes):
        coords[i] = np.vstack([host_pos, xyz_subset[i].astype(np.float32)])
        boxes[i] = box_single

    np.save(set_dir / "coord.npy", coords)
    np.save(set_dir / "box.npy", boxes)
    np.save(set_dir / "energy.npy", energies)

    atom_types = np.asarray([type_idx[s] for s in syms], dtype=np.int32)
    atom_types.tofile(set_dir / "type.raw")

    (system_dir / "type_map.raw").write_text("\n".join(TYPE_MAP) + "\n", encoding="utf-8")

def load_coronene_arr4(path, n_use=None):
    arr4 = as_xyze(np.load(path))
    if n_use is not None and n_use < arr4.shape[0]:
        arr4 = arr4[:n_use]
    return arr4

def main():
    root = Path(OUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    data = np.load(MULTI_NPY, allow_pickle=True).item()

    d1 = as_xyze(np.array(data["C_Coronene"][MULTI_METHOD]))
    d2 = as_xyze(np.array(data["CC_Coronene"][MULTI_METHOD]))
    d3 = as_xyze(np.array(data["CCC_Coronene"][MULTI_METHOD]))
    d0 = load_coronene_arr4(CORONENE_NPY, n_use=CORONENE_N_USE)

    systems = [
        ("system_ring1", 1, d0),
        ("system_ring2", 2, d1),
        ("system_ring3", 3, d2),
        ("system_ring4", 4, d3),
    ]

    for name, n_ring, arr4 in systems:
        arr4 = np.asarray(arr4, dtype=float)
        before = arr4.shape[0]

        arr4 = filter_by_energy_cm1(arr4)
        after_energy = arr4.shape[0]

        arr4 = remove_forbidden_points(arr4, tol=0.1)
        after_forbidden = arr4.shape[0]

        train4, val4, independent_val4, train_groups, val_groups, independent_val_groups = split_by_cross_block(
            arr4,
            seed=SEED + 1000 * n_ring,
            max_frac=MAX_FRAC,
            train_ratio=TRAIN_RATIO
        )

        xyz_train = train4[:, :3]
        E_train_ev = train4[:, 3] / CM1_PER_EV

        xyz_val = val4[:, :3]
        E_val_ev = val4[:, 3] / CM1_PER_EV

        if n_ring == 1:
            host_syms, host_pos = CORONENE_SYMS, CORONENE_POS
        else:
            host_syms, host_pos = make_hex(n_ring)

        system_dir = root / name

        write_system(system_dir, "set.000", host_syms, host_pos, xyz_train, E_train_ev)
        write_system(system_dir, "set.001", host_syms, host_pos, xyz_val, E_val_ev)

        emax_written_ev = max(
            np.max(np.abs(E_train_ev)) if E_train_ev.size else 0.0,
            np.max(np.abs(E_val_ev)) if E_val_ev.size else 0.0
        )

        print(
            f"[OK] {name}: ring={n_ring} "
            f"energy_filter {before}->{after_energy}, "
            f"remove_forbidden {after_energy}->{after_forbidden}, "
            f"train={xyz_train.shape[0]} val={xyz_val.shape[0]} "
            f"groups_train={len(train_groups)} groups_val={len(val_groups)} "
            f"nat={len(host_syms)+1} "
            f"max|E|={emax_written_ev:.8f} eV "
            f"({emax_written_ev * CM1_PER_EV:.3f} cm^-1)"
        )

    print(f"[DONE] written to {OUT_ROOT}/system_ring1..4")

if __name__ == "__main__":
    main()