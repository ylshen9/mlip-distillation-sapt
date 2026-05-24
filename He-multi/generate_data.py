import os, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write

#from orb_models.forcefield import pretrained
#from orb_models.forcefield.calculator import ORBCalculator
from mattersim.forcefield import MatterSimCalculator

OUT_ROOT = "dataset"
TRAIN_SET = "set.000"
VAL_SET = "set.001"

N_TRAIN_PER = 600
N_VAL_PER = 200

BOX_SIZE = 40.0
SEED = 2025

R_MIN = 1.0
CORE_RANGE = (2.0, 6.0)
TAIL_RANGE = (6.0, 12.0)
CORE_WEIGHT = 0.8
Z_BIAS = 0.0

SAVE_POSCARS = True

DEFAULT_CHARGE = 0
DEFAULT_MULT = 1

E_MAX_CM1 = 1000.0
CM1_PER_EV = 8065.544005

MAX_TOTAL_SAMPLES = 50000

MULTI_NPY = "multi.npy"
MULTI_METHOD = "pbe0_def2-SVP"

CORONENE_NPY = "coronene.npy"
CORONENE_N_USE = None

rng = np.random.default_rng(SEED)

BZ_POS_HEX = np.array([
    [-1.2073830,-0.6970829,0.0],[-1.2073830,0.6970829,0.0],[0.0,1.3941659,0.0],
    [ 1.2073830,0.6970829,0.0],[ 1.2073830,-0.6970829,0.0],[0.0,-1.3941659,0.0],
    [-2.1490090,-1.2407309,0.0],[-2.1490090,1.2407309,0.0],[0.0,2.4814619,0.0],
    [ 2.1490090,1.2407309,0.0],[ 2.1490090,-1.2407309,0.0],[0.0,-2.4814619,0.0],
], dtype=float)

BZ_C = BZ_POS_HEX[:6, :2]
BZ_H = BZ_POS_HEX[6:, :2]
R_C = np.linalg.norm(BZ_C, axis=1).mean()
R_H = np.linalg.norm(BZ_H, axis=1).mean()
R_CH = R_H - R_C
L = R_C * math.sqrt(3.0)
a1 = np.array([L, 0.0], dtype=float)
a2 = np.array([L * 0.5, L * math.sqrt(3) / 2], dtype=float)

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
            dx = R_C * math.cos(ang)
            dy = R_C * math.sin(ang)
            C_list.append([cx + dx, cy + dy])

    def dedup(points, tol=1e-4):
        table = {}
        for x, y in points:
            key = (round(x / tol) * tol, round(y / tol) * tol)
            table.setdefault(key, [x, y])
        return np.array(list(table.values()), dtype=float)

    C_2d = dedup(C_list)
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
                v_sum += (C_2d[j] - C_2d[i])
            out = -v_sum
            out /= np.linalg.norm(out)
            H_list.append(C_2d[i] + out * R_CH)
    H_2d = np.array(H_list, dtype=float)

    C_3d = np.column_stack([C_2d, np.zeros(len(C_2d))]) + shift
    H_3d = np.column_stack([H_2d, np.zeros(len(H_2d))]) + shift
    syms = ["C"] * len(C_3d) + ["H"] * len(H_3d)
    pos = np.vstack([C_3d, H_3d])
    return syms, pos

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

def as_xyze(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Expected array shape (M,4+) with columns [x,y,z,E].")
    return arr[:, :4].astype(float)

def load_multi_dirs_hex(multi_path, method, host_center):
    data = np.load(multi_path, allow_pickle=True).item()
    ring_key = {2: "C_Coronene", 3: "CC_Coronene", 4: "CCC_Coronene"}
    dirs = {}
    for ring, key in ring_key.items():
        if key not in data or method not in data[key]:
            raise KeyError(f"Missing data['{key}']['{method}'] in {multi_path}.")
        arr4 = as_xyze(np.array(data[key][method], dtype=float))
        xyz = arr4[:, :3]
        rel = xyz - host_center[None, :]
        r = np.linalg.norm(rel, axis=1)
        mask = r > 1e-8
        u = rel[mask] / r[mask, None]
        dirs[ring] = u.astype(np.float64)
    return dirs

def load_coronene_dirs(npy_path, host_center, n_use=None):
    arr = as_xyze(np.load(npy_path))
    if n_use is not None and n_use < len(arr):
        arr = arr[:n_use]
    xyz = arr[:, :3].astype(float)
    rel = xyz - host_center[None, :]
    r = np.linalg.norm(rel, axis=1)
    mask = r > 1e-8
    u = rel[mask] / r[mask, None]
    return u.astype(np.float64)

def sample_radius_vol_uniform(rmin, rmax):
    u = rng.uniform(rmin**3, rmax**3)
    return u ** (1.0 / 3.0)

def sample_r_with_preference():
    if rng.random() < CORE_WEIGHT:
        a, b = CORE_RANGE
    else:
        a, b = TAIL_RANGE
    return sample_radius_vol_uniform(a, b)

def rot_z(u, ang):
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = u
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float)

def place_he_c6_from_multi_dirs(host, multi_dirs, z_bias=0.0, r_min=2.5, max_trial=400):
    center = host.get_positions().mean(axis=0)
    host_pos = host.get_positions()
    for _ in range(max_trial):
        r = sample_r_with_preference()
        u = multi_dirs[rng.integers(0, len(multi_dirs))]
        k = int(rng.integers(0, 6))
        u = rot_z(u, k * (np.pi / 3.0))
        pos = center + r * u + np.array([0.0, 0.0, z_bias], dtype=float)
        dists = np.linalg.norm(host_pos - pos, axis=1)
        if np.all(dists >= r_min):
            return pos
    raise RuntimeError("Sampling failed: too tight constraints.")

def atoms_to_type_indices(symbols, type_map):
    idx = [type_map.index(s) for s in symbols]
    return np.array(idx, dtype=np.int32)

def label_orb(at, calc):
    at.calc = calc
    e = at.get_potential_energy()
    f = at.get_forces()
    return e, f

def write_deepmd_set(out_dir, coords, boxes, forces, energies, type_raw, type_map, save_poscars=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coord.npy", np.array(coords, dtype=np.float64))
    np.save(out_dir / "box.npy", np.array(boxes, dtype=np.float64))
    np.save(out_dir / "force.npy", np.array(forces, dtype=np.float64))
    np.save(out_dir / "energy.npy", np.array(energies, dtype=np.float64))
    (out_dir / "type.raw").write_bytes(type_raw.tobytes())
    (out_dir.parent / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    if save_poscars:
        pos_dir = out_dir / "poscars"
        pos_dir.mkdir(parents=True, exist_ok=True)
        for i, (xyz, box) in enumerate(zip(coords, boxes)):
            syms = [type_map[int(k)] for k in type_raw]
            at = Atoms(symbols=syms, positions=xyz, pbc=False)
            at.set_cell(box)
            write(pos_dir / f"POSCAR_{i:06d}.vasp", at, direct=False, vasp5=True)

def build_host_from_ring(n_ring):
    if n_ring == 1:
        host = Atoms(symbols=CORONENE_SYMS, positions=CORONENE_POS, pbc=False)
    else:
        syms, pos = make_hex(n_ring)
        host = Atoms(symbols=syms, positions=pos, pbc=False)
    host.set_pbc(False)
    host.set_cell(np.eye(3) * BOX_SIZE)
    host.info["charge"] = DEFAULT_CHARGE
    host.info["spin_multiplicity"] = DEFAULT_MULT
    return host

def compute_reference_energy(host, calc, far_z=12.0):
    center = host.get_positions().mean(axis=0)
    he_far = center + np.array([0.0, 0.0, far_z], dtype=float)
    base_symbols = host.get_chemical_symbols() + ["He"]
    ref_positions = np.vstack([host.get_positions(), he_far[None, :]])
    at_ref = Atoms(symbols=base_symbols, positions=ref_positions, pbc=False)
    at_ref.set_cell(np.eye(3) * BOX_SIZE)
    at_ref.info["charge"] = DEFAULT_CHARGE
    at_ref.info["spin_multiplicity"] = DEFAULT_MULT
    e_ref, _ = label_orb(at_ref, calc)
    return float(e_ref)

def generate_system_dataset(system_dir, n_ring, calc, multi_dirs, source_meta):
    system_dir.mkdir(parents=True, exist_ok=True)
    host = build_host_from_ring(n_ring)
    base_symbols = host.get_chemical_symbols() + ["He"]
    type_map = ["C", "H", "He"]
    type_row = atoms_to_type_indices(base_symbols, type_map)

    e_ref = compute_reference_energy(host, calc, far_z=20.0)
    print(f"[ring={n_ring}] E_ref={e_ref:.6f} eV host_atoms={len(host)} total_atoms={len(base_symbols)} dirs={len(multi_dirs)}")

    n_target = N_TRAIN_PER + N_VAL_PER
    coords_all, boxes_all, forces_all, etot_all = [], [], [], []
    n_generated = 0

    while True:
        if n_generated >= MAX_TOTAL_SAMPLES:
            raise RuntimeError(
                f"[ring={n_ring}] Reached MAX_TOTAL_SAMPLES={MAX_TOTAL_SAMPLES} but only collected fewer than {n_target} valid samples. "
                f"Consider increasing MAX_TOTAL_SAMPLES or relaxing E_MAX_CM1."
            )

        he_pos = place_he_c6_from_multi_dirs(
            host,
            multi_dirs=multi_dirs,
            z_bias=Z_BIAS,
            r_min=R_MIN,
            max_trial=400
        )
        positions = np.vstack([host.get_positions(), he_pos[None, :]])

        at = Atoms(symbols=base_symbols, positions=positions, pbc=False)
        at.set_cell(np.eye(3) * BOX_SIZE)
        #Add these two when using ORB
        #at.info["charge"] = DEFAULT_CHARGE
        #at.info["spin_multiplicity"] = DEFAULT_MULT

        e_tot, f = label_orb(at, calc)

        coords_all.append(at.get_positions())
        boxes_all.append(at.cell.array)
        forces_all.append(f)
        etot_all.append(e_tot)

        n_generated += 1

        etot_arr = np.array(etot_all, dtype=np.float64)
        ebind_ev = etot_arr - e_ref
        good_mask = np.abs(ebind_ev * CM1_PER_EV) < E_MAX_CM1
        n_good = int(good_mask.sum())

        if n_generated % 100 == 0:
            print(f"[ring={n_ring}] generated={n_generated} good={n_good} last_Ebind={(e_tot - e_ref):.6f} eV")

        if n_good >= n_target:
            print(f"[ring={n_ring}] reached target good={n_good} total_generated={n_generated}")
            break

    coords_all = np.array(coords_all, dtype=np.float64)
    boxes_all = np.array(boxes_all, dtype=np.float64)
    forces_all = np.array(forces_all, dtype=np.float64)
    etot_all = np.array(etot_all, dtype=np.float64)

    ebind_all = etot_all - e_ref
    good_mask = np.abs(ebind_all * CM1_PER_EV) < E_MAX_CM1
    good_idx = np.nonzero(good_mask)[0]

    if good_idx.size < n_target:
        raise RuntimeError(f"[ring={n_ring}] Internal error: good_idx.size={good_idx.size} < target={n_target}")

    perm = rng.permutation(good_idx.size)
    sel = good_idx[perm[:n_target]]

    coords_sel = coords_all[sel]
    boxes_sel = boxes_all[sel]
    forces_sel = forces_all[sel]
    energies_sel = ebind_all[sel]

    train_idx = np.arange(N_TRAIN_PER)
    val_idx = np.arange(N_TRAIN_PER, n_target)

    write_deepmd_set(system_dir / TRAIN_SET, coords_sel[train_idx], boxes_sel[train_idx], forces_sel[train_idx], energies_sel[train_idx], type_row, type_map, save_poscars=SAVE_POSCARS)
    write_deepmd_set(system_dir / VAL_SET, coords_sel[val_idx], boxes_sel[val_idx], forces_sel[val_idx], energies_sel[val_idx], type_row, type_map, save_poscars=SAVE_POSCARS)

    meta = {
        "ring": int(n_ring),
        "n_train": int(N_TRAIN_PER),
        "n_val": int(N_VAL_PER),
        "box_size": float(BOX_SIZE),
        "core_range_A": [float(CORE_RANGE[0]), float(CORE_RANGE[1])],
        "tail_range_A": [float(TAIL_RANGE[0]), float(TAIL_RANGE[1])],
        "core_weight": float(CORE_WEIGHT),
        "r_min_avoid_A": float(R_MIN),
        "z_bias_A": float(Z_BIAS),
        "charge": int(DEFAULT_CHARGE),
        "spin_multiplicity": int(DEFAULT_MULT),
        "energy_reference_eV": float(e_ref),
        "energy_max_abs_bind_cm1": float(E_MAX_CM1),
        "cm1_per_eV": float(CM1_PER_EV),
        "n_generated_total": int(n_generated),
        "n_good_total": int(good_idx.size),
        "label": "binding_energy",
        "label_unit": "eV",
        "n_atoms_host": int(len(host)),
        "n_atoms_total_with_He": int(len(base_symbols)),
        "direction_source": dict(source_meta),
    }
    (system_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK][ring={n_ring}] wrote {str(system_dir)}")

def main():
    #orbff = pretrained.orb_v3_conservative_omol(device="cpu", precision="float32-high")
    #calc = ORBCalculator(orbff, device="cpu")
    calc = MatterSimCalculator(device='cpu')

    root = Path(OUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    multi_dirs_hex_all = {}
    for n_ring in (2, 3, 4):
        host = build_host_from_ring(n_ring)
        center = host.get_positions().mean(axis=0)
        dirs_map = load_multi_dirs_hex(MULTI_NPY, MULTI_METHOD, host_center=center)
        multi_dirs_hex_all[n_ring] = dirs_map[n_ring]
        print(f"[dirs] ring={n_ring} n_dirs={len(multi_dirs_hex_all[n_ring])}")

    host1 = build_host_from_ring(1)
    center1 = host1.get_positions().mean(axis=0)
    dirs1 = load_coronene_dirs(CORONENE_NPY, host_center=center1, n_use=CORONENE_N_USE)
    print(f"[dirs] ring=1 n_dirs={len(dirs1)} n_use={CORONENE_N_USE}")

    rings = (1, 2, 3, 4)
    for n_ring in rings:
        system_dir = root / f"system_ring{n_ring}"
        if n_ring == 1:
            dirs = dirs1
            source_meta = {"file": str(CORONENE_NPY), "format": "npy_array_xyzE", "n_use": CORONENE_N_USE}
        else:
            dirs = multi_dirs_hex_all[n_ring]
            source_meta = {"file": str(MULTI_NPY), "method": str(MULTI_METHOD), "format": "dict[key][method]->xyzE"}
        generate_system_dataset(system_dir, n_ring, calc, multi_dirs=dirs, source_meta=source_meta)

    meta_all = {
        "systems": [f"system_ring{r}" for r in rings],
        "n_train_per_system": int(N_TRAIN_PER),
        "n_val_per_system": int(N_VAL_PER),
        "n_train_total": int(len(rings) * N_TRAIN_PER),
        "n_val_total": int(len(rings) * N_VAL_PER),
        "seed": int(SEED),
        "label": "binding_energy",
        "label_unit": "eV",
        "ring1_source": {"file": str(CORONENE_NPY), "n_use": CORONENE_N_USE},
        "rings234_source": {"file": str(MULTI_NPY), "method": str(MULTI_METHOD)},
    }
    (root / "meta_all.json").write_text(json.dumps(meta_all, indent=2), encoding="utf-8")
    print(f"[DONE] dataset written to: {OUT_ROOT}")

if __name__ == "__main__":
    main()
