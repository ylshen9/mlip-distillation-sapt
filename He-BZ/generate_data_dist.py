import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
import json
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

OUT_ROOT = "dataset1"
TRAIN_SET = "set.000"
VAL_SET = "set.001"

N_TRAIN = 2400
N_VAL = 800

BOX_SIZE = 20.0
SEED = 2025

R_SAMPLE_MIN = 2.5
R_SAMPLE_MAX = 6.0
R_MIN = 1.0
Z_BIAS = 0.0

SAVE_POSCARS = True

DEFAULT_CHARGE = 0
DEFAULT_MULT = 1

E_MAX_CM1 = 1000.0
CM1_PER_EV = 8065.544005

MAX_TOTAL_SAMPLES = 50000

rng = np.random.default_rng(SEED)


def make_benzene():
    coords = np.array([
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
    symbols = ["C"] * 6 + ["H"] * 6
    return Atoms(symbols=symbols, positions=coords, pbc=False)


def sample_he_position(center, host_pos):
    for _ in range(400):
        u = rng.uniform(R_SAMPLE_MIN**3, R_SAMPLE_MAX**3)
        r = u ** (1.0 / 3.0)
        cos_theta = rng.uniform(-1.0, 1.0)
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
        phi = rng.uniform(0.0, 2 * np.pi)
        direction = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
        pos = center + r * direction + np.array([0.0, 0.0, Z_BIAS])
        if np.all(np.linalg.norm(host_pos - pos, axis=1) >= R_MIN):
            return pos
    raise RuntimeError("He placement failed.")


def atoms_to_type_indices(symbols, type_map):
    return np.array([type_map.index(s) for s in symbols], dtype=np.int32)


def label_orb(at, calc):
    at.info["charge"] = DEFAULT_CHARGE
    at.info["spin"] = DEFAULT_MULT
    at.info["spin_multiplicity"] = DEFAULT_MULT
    at.calc = calc
    return at.get_potential_energy(), at.get_forces()


def write_deepmd_set(out_dir, coords, boxes, forces, energies, type_raw, type_map):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coord.npy",  np.array(coords,   dtype=np.float64))
    np.save(out_dir / "box.npy",    np.array(boxes,    dtype=np.float64))
    np.save(out_dir / "force.npy",  np.array(forces,   dtype=np.float64))
    np.save(out_dir / "energy.npy", np.array(energies, dtype=np.float64))
    np.array(type_raw, dtype=np.int32).tofile(out_dir / "type.raw")
    (out_dir.parent / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")

    if SAVE_POSCARS:
        pos_dir = out_dir / "poscars"
        pos_dir.mkdir(parents=True, exist_ok=True)
        for i, (xyz, box) in enumerate(zip(coords, boxes)):
            at = Atoms(symbols=[type_map[int(k)] for k in type_raw], positions=xyz, pbc=False)
            at.set_cell(box)
            at.center()
            write(pos_dir / f"POSCAR_{i:06d}.vasp", at, direct=False, vasp5=True)


def main():
    orbff = pretrained.orb_v3_conservative_omol(device="cpu", precision="float32-high")
    calc = ORBCalculator(orbff, device="cpu")

    benz = make_benzene()
    benz.set_pbc(False)
    benz.set_cell(np.eye(3) * BOX_SIZE)
    benz.center()
    benz.info["charge"] = DEFAULT_CHARGE
    benz.info["spin_multiplicity"] = DEFAULT_MULT

    type_map = ["C", "H", "He"]
    base_symbols = benz.get_chemical_symbols() + ["He"]
    type_row = atoms_to_type_indices(base_symbols, type_map)
    center = benz.get_positions().mean(axis=0)

    he_far = center + np.array([0.0, 0.0, 12.0])
    at_ref = Atoms(symbols=base_symbols,
                   positions=np.vstack([benz.get_positions(), he_far[None, :]]),
                   pbc=False)
    at_ref.set_cell(np.eye(3) * BOX_SIZE)
    at_ref.center()
    E_ref, _ = label_orb(at_ref, calc)

    N_TARGET = N_TRAIN + N_VAL
    all_coords, all_boxes, all_forces, all_E_tot = [], [], [], []
    n_generated = 0

    while True:
        if n_generated >= MAX_TOTAL_SAMPLES:
            raise RuntimeError("Reached MAX_TOTAL_SAMPLES.")

        he_pos = sample_he_position(center, benz.get_positions())
        at = Atoms(symbols=base_symbols,
                   positions=np.vstack([benz.get_positions(), he_pos[None, :]]),
                   pbc=False)
        at.set_cell(np.eye(3) * BOX_SIZE)
        at.center()

        e_tot, f = label_orb(at, calc)
        all_coords.append(at.get_positions())
        all_boxes.append(at.cell.array)
        all_forces.append(f)
        all_E_tot.append(e_tot)
        n_generated += 1

        E_bind_eV = np.array(all_E_tot) - E_ref
        good_mask = np.abs(E_bind_eV * CM1_PER_EV) < E_MAX_CM1
        n_good = int(good_mask.sum())

        if n_generated % 100 == 0:
            print(f"generated={n_generated}  good={n_good}  "
                  f"last_E_bind={( e_tot - E_ref) * CM1_PER_EV:.4f} cm-1")

        if n_good >= N_TARGET:
            break

    all_coords  = np.array(all_coords,  dtype=np.float64)
    all_boxes   = np.array(all_boxes,   dtype=np.float64)
    all_forces  = np.array(all_forces,  dtype=np.float64)
    all_E_tot   = np.array(all_E_tot,   dtype=np.float64)

    E_bind_eV = all_E_tot - E_ref
    good_idx  = np.nonzero(np.abs(E_bind_eV * CM1_PER_EV) < E_MAX_CM1)[0]
    sel = good_idx[rng.permutation(len(good_idx))[:N_TARGET]]

    coords_sel   = all_coords[sel]
    boxes_sel    = all_boxes[sel]
    forces_sel   = all_forces[sel]
    energies_sel = E_bind_eV[sel]

    root = Path(OUT_ROOT)
    write_deepmd_set(root / TRAIN_SET, coords_sel[:N_TRAIN], boxes_sel[:N_TRAIN],
                     forces_sel[:N_TRAIN], energies_sel[:N_TRAIN], type_row, type_map)
    write_deepmd_set(root / VAL_SET,   coords_sel[N_TRAIN:], boxes_sel[N_TRAIN:],
                     forces_sel[N_TRAIN:], energies_sel[N_TRAIN:], type_row, type_map)

    meta = {
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "r_sample_min_A": R_SAMPLE_MIN,
        "r_sample_max_A": R_SAMPLE_MAX,
        "r_hard_min_A": R_MIN,
        "z_bias": Z_BIAS,
        "box_size": BOX_SIZE,
        "charge": DEFAULT_CHARGE,
        "spin_multiplicity": DEFAULT_MULT,
        "energy_reference_eV": float(E_ref),
        "energy_max_abs_bind_cm1": E_MAX_CM1,
        "cm1_per_eV": CM1_PER_EV,
        "n_generated_total": n_generated,
        "n_good_total": int(len(good_idx)),
        "label": "binding_energy",
        "label_unit": "eV"
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Written to {OUT_ROOT}/{{{TRAIN_SET},{VAL_SET}}}")


if __name__ == "__main__":
    main()