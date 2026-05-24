import numpy as np
from pathlib import Path

NPY_PATH = "./HeBz_sector_2475.npy"
OUT_ROOT = "data"
CELL_A = 20.0
CM1_PER_eV = 8065.544
SEED = 42
MAX_FRAC = 0.8

BZ_SYMS = ["C"] * 6 + ["H"] * 6
BZ_POS = np.array([
    [-1.2073830, -0.6970829, 0.0],
    [-1.2073830,  0.6970829, 0.0],
    [ 0.0000000,  1.3941659, 0.0],
    [ 1.2073830,  0.6970829, 0.0],
    [ 1.2073830, -0.6970829, 0.0],
    [ 0.0000000, -1.3941659, 0.0],
    [-2.1490090, -1.2407309, 0.0],
    [-2.1490090,  1.2407309, 0.0],
    [ 0.0000000,  2.4814619, 0.0],
    [ 2.1490090,  1.2407309, 0.0],
    [ 2.1490090, -1.2407309, 0.0],
    [ 0.0000000, -2.4814619, 0.0],
], dtype=float)

TYPE_MAP = ["C", "H", "He"]
type_idx = {"C": 0, "H": 1, "He": 2}

SYMS = BZ_SYMS + ["He"]
POS0 = BZ_POS


def write_type_raw(path: Path, types):
    np.array(types, dtype=np.int32).tofile(path)


def write_system(subdir: Path, xyz_subset: np.ndarray, energy_subset_eV: np.ndarray):
    nat = len(SYMS)
    nframes = len(xyz_subset)

    coord = np.zeros((nframes, nat * 3), dtype=np.float32)
    box = np.zeros((nframes, 9), dtype=np.float32)
    ener = np.array(energy_subset_eV, dtype=np.float64)

    box_single = np.diag([CELL_A, CELL_A, CELL_A]).reshape(-1)

    for i, (x, y, z) in enumerate(xyz_subset):
        pos = np.vstack([POS0, [x, y, z]])
        coord[i, :] = pos.reshape(-1)
        box[i, :] = box_single

    subdir.mkdir(parents=True, exist_ok=True)
    np.save(subdir / "coord.npy", coord)
    np.save(subdir / "box.npy", box)
    np.save(subdir / "energy.npy", ener)

    atom_types = [type_idx[s] for s in SYMS]
    write_type_raw(subdir / "type.raw", atom_types)


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

    groups = (
        z_bin * (n_r * n_theta)
        + r_bin * n_theta
        + theta_bin
    ).astype(int)

    return groups


def ordered_unique(arr):
    seen = set()
    out = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return np.array(out, dtype=arr.dtype)


def select_first_groups(xyz, energy, groups, ordered_groups, frac):
    n_groups_total = len(ordered_groups)
    n_take = int(n_groups_total * frac)

    if frac > 0 and n_take < 1:
        n_take = 1
    if frac >= 1.0:
        n_take = n_groups_total

    chosen_groups = ordered_groups[:n_take]
    mask = np.isin(groups, chosen_groups)

    return xyz[mask], energy[mask], groups[mask], chosen_groups


def main():
    assert 0.0 < MAX_FRAC <= 1.0, "MAX_FRAC must be in (0, 1]"

    arr = np.load(NPY_PATH)
    if arr.shape[1] < 4:
        raise ValueError("NPY must have at least 4 columns: x y z Eb_ref_cm^-1")

    xyz = arr[:, :3].astype(float)
    Eb_ref_cm1 = arr[:, 3].astype(float)

    mask = np.abs(Eb_ref_cm1) < 1000.0
    xyz = xyz[mask]
    Eb_ref_cm1 = Eb_ref_cm1[mask]

    if len(xyz) == 0:
        raise RuntimeError("No samples after masking.")

    rng = np.random.default_rng(SEED)

    groups = make_groups(xyz, n_r=8, n_theta=6)

    unique_groups = np.unique(groups)
    shuffled_groups = rng.permutation(unique_groups)

    n_group_pool = int(len(shuffled_groups) * MAX_FRAC)
    if n_group_pool < 1:
        raise RuntimeError("Too few groups for pool split.")

    groups_pool_order = shuffled_groups[:n_group_pool]
    groups_validation_order = shuffled_groups[n_group_pool:]

    pool_mask = np.isin(groups, groups_pool_order)
    validation_mask = np.isin(groups, groups_validation_order)

    xyz_pool = xyz[pool_mask]
    Eb_pool_cm1 = Eb_ref_cm1[pool_mask]
    groups_pool = groups[pool_mask]

    xyz_validation = xyz[validation_mask]
    Eb_validation_cm1 = Eb_ref_cm1[validation_mask]

    if len(xyz_validation) == 0:
        raise RuntimeError("Validation set is empty after group split.")

    root = Path(OUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    validation_arr = np.column_stack([xyz_validation, Eb_validation_cm1])
    np.save(root / "validation.npy", validation_arr)


    max_pct = int(round(MAX_FRAC * 100))
    fractions = [0.1 * i / MAX_FRAC for i in range(1, max_pct // 10 + 1)]

    groups_pool_order = ordered_unique(groups_pool_order)

    for i, frac in enumerate(fractions, start=1):
        xyz_subset, Eb_subset_cm1, groups_subset, groups_subset_order = select_first_groups(
            xyz_pool, Eb_pool_cm1, groups_pool, groups_pool_order, frac
        )

        n_use = len(xyz_subset)
        if n_use < 2:
            raise RuntimeError(f"dataset{i} would contain too few samples for train/val split.")

        subset_group_order = ordered_unique(groups_subset_order)
        n_train_groups = int(len(subset_group_order) * 0.8)

        if n_train_groups < 1:
            n_train_groups = 1
        if n_train_groups >= len(subset_group_order):
            n_train_groups = len(subset_group_order) - 1
        if n_train_groups < 1:
            raise RuntimeError(f"dataset{i} has too few groups for train/val split.")

        train_groups = subset_group_order[:n_train_groups]
        val_groups = subset_group_order[n_train_groups:]

        train_mask = np.isin(groups_subset, train_groups)
        val_mask = np.isin(groups_subset, val_groups)

        xyz_train = xyz_subset[train_mask]
        Eb_train_eV = Eb_subset_cm1[train_mask] / CM1_PER_eV

        xyz_val = xyz_subset[val_mask]
        Eb_val_eV = Eb_subset_cm1[val_mask] / CM1_PER_eV

        if len(xyz_train) < 1 or len(xyz_val) < 1:
            raise RuntimeError(
                f"dataset{i} split invalid: train={len(xyz_train)}, val={len(xyz_val)}"
            )

        dataset_root = root / f"dataset{i}"
        dataset_root.mkdir(parents=True, exist_ok=True)

        with open(dataset_root / "type_map.raw", "w") as f:
            for s in TYPE_MAP:
                f.write(f"{s}\n")

        write_system(dataset_root / "set.000", xyz_train, Eb_train_eV)
        write_system(dataset_root / "set.001", xyz_val, Eb_val_eV)

        actual_pct = int(round(frac * MAX_FRAC * 100))
        print(
            f"[OK] {dataset_root} ({actual_pct}% of total): total={n_use}, "
            f"train={len(xyz_train)}, val={len(xyz_val)}, "
            f"groups_total={len(subset_group_order)}, "
            f"groups_train={len(train_groups)}, groups_val={len(val_groups)}"
        )

    print(f"[OK] independent validation saved to: {root / 'validation.npy'} "
          f"({int(round((1 - MAX_FRAC) * 100))}% of total, size={len(validation_arr)})")


if __name__ == "__main__":
    main()