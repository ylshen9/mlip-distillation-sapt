import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_type_map(type_map_path: Path):
    txt = type_map_path.read_text(encoding="utf-8").strip().splitlines()
    return [t.strip() for t in txt if t.strip()]


def load_type_raw(type_raw_path: Path, n_atoms: int):
    arr = np.fromfile(type_raw_path, dtype=np.int32)
    assert arr.size == n_atoms, f"type.raw atom count {arr.size} != expected {n_atoms}"
    return arr


def compute_r_min_numpy(coords: np.ndarray, he_index: int) -> np.ndarray:
    he_pos = coords[:, he_index:he_index + 1, :]
    diff = he_pos - coords
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.ones_like(dist, dtype=bool)
    mask[:, he_index] = False
    dist_masked = np.where(mask, dist, np.inf)
    r_min = dist_masked.min(axis=-1)
    return r_min


class SystemSetDataset(Dataset):
    def __init__(self, system_dir: str, set_name: str = "set.000"):
        self.system_dir = Path(system_dir)
        self.set_dir = self.system_dir / set_name

        coords = np.load(self.set_dir / "coord.npy")
        boxes = np.load(self.set_dir / "box.npy")
        energies = np.load(self.set_dir / "energy.npy")

        if coords.ndim == 3:
            M, N, D = coords.shape
            assert D == 3
        elif coords.ndim == 2:
            M, D = coords.shape
            assert D % 3 == 0
            N = D // 3
            coords = coords.reshape(M, N, 3)
        else:
            raise ValueError("coord.npy must be (M,N,3) or (M,3N)")

        if boxes.ndim == 3:
            pass
        elif boxes.ndim == 2 and boxes.shape[1] == 9:
            boxes = boxes.reshape(-1, 3, 3)
        else:
            boxes = boxes.reshape(1, 3, 3)

        if energies.ndim == 0:
            energies = energies[None]
        assert energies.ndim == 1

        self.coords = coords.astype(np.float32)
        self.boxes = boxes.astype(np.float32)
        self.energies = energies.astype(np.float32)
        self.M, self.N, _ = self.coords.shape

        type_map = load_type_map(self.system_dir / "type_map.raw")
        type_raw = load_type_raw(self.set_dir / "type.raw", self.N)
        self.type_map = type_map
        self.type_raw = type_raw.astype(np.int64)

        assert "He" in type_map
        self.he_type_id = int(type_map.index("He"))
        he_indices = np.where(type_raw == self.he_type_id)[0]
        assert he_indices.size == 1
        self.he_index_local = int(he_indices[0])

        self.r_min = compute_r_min_numpy(self.coords, self.he_index_local).astype(np.float32)

        force_path = self.set_dir / "force.npy"
        if force_path.is_file():
            forces = np.load(force_path)
            if forces.ndim == 2 and forces.shape[1] == 3 * self.N:
                forces = forces.reshape(self.M, self.N, 3)
            assert forces.shape == (self.M, self.N, 3)
            self.forces = forces.astype(np.float32)
            self.has_forces = True
        else:
            self.forces = None
            self.has_forces = False

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        Z = torch.from_numpy(self.type_raw)
        pos = torch.from_numpy(self.coords[idx])
        E = torch.tensor(self.energies[idx], dtype=torch.float32)
        r_min = torch.tensor(self.r_min[idx], dtype=torch.float32)
        if self.has_forces:
            Fv = torch.from_numpy(self.forces[idx])
        else:
            Fv = torch.zeros(self.N, 3, dtype=torch.float32)

        if self.boxes.shape[0] == 1:
            box_np = self.boxes[0]
        else:
            box_np = self.boxes[idx]
        box = torch.from_numpy(box_np.astype(np.float32))
        return Z, pos, E, Fv, r_min, box


def find_he_pos(Z: torch.Tensor, pos: torch.Tensor, he_type_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N = Z.shape
    mask = (Z == he_type_id)
    if not torch.all(mask.sum(dim=1) == 1):
        raise RuntimeError("each sample must contain exactly one He atom")
    he_index = mask.long().argmax(dim=1)
    he_pos = pos[torch.arange(B, device=pos.device), he_index].unsqueeze(1)
    return he_pos, he_index


class MultiShellDescriptor(nn.Module):
    def __init__(self, he_type_id: int, n_shells: int = 2, n_rbf_per_shell: int = 16, r_cuts=None):
        super().__init__()
        self.he_type_id = int(he_type_id)
        self.n_shells = n_shells
        self.n_rbf_per_shell = n_rbf_per_shell
        if r_cuts is None:
            r_cuts = [3, 12.0]
        assert len(r_cuts) == n_shells
        self.r_cuts = nn.Parameter(torch.tensor(r_cuts, dtype=torch.float32), requires_grad=False)

        centers_list = []
        gamma_list = []
        for s in range(n_shells):
            r_cut = float(r_cuts[s])
            centers = torch.linspace(0.0, r_cut, n_rbf_per_shell)
            dr = r_cut / max(n_rbf_per_shell, 1)
            gamma = 1.0 / (2.0 * dr * dr) if dr > 0 else 1.0
            centers_list.append(centers)
            gamma_list.append(gamma)
        centers_all = torch.stack(centers_list, dim=0)
        gamma_all = torch.tensor(gamma_list, dtype=torch.float32)
        self.register_buffer("rbf_centers", centers_all)
        self.register_buffer("rbf_gamma", gamma_all)

    def forward(self, Z: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        he_pos, _ = find_he_pos(Z, pos, self.he_type_id)
        diff = pos - he_pos
        r = torch.norm(diff, dim=-1)
        feats = []
        for s in range(self.n_shells):
            r_cut = self.r_cuts[s]
            centers = self.rbf_centers[s].view(1, 1, -1)
            gamma = self.rbf_gamma[s]
            rc = torch.clamp(r, max=r_cut)
            fc = 0.5 * (torch.cos(rc * math.pi / r_cut) + 1.0)
            fc = fc * (r < r_cut).float()
            r_exp = r.unsqueeze(-1)
            rbf = torch.exp(-gamma * (r_exp - centers) ** 2) * fc.unsqueeze(-1)
            feats.append(rbf)
        return torch.cat(feats, dim=-1)


class KSpaceSOGMultiLayer(nn.Module):
    def __init__(self, n_layers: int = 4, n_gaussians: int = 8, nmax: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.n_gaussians = n_gaussians

        h_list = []
        for n1 in range(-nmax, nmax + 1):
            for n2 in range(-nmax, nmax + 1):
                for n3 in range(-nmax, nmax + 1):
                    if n1 == 0 and n2 == 0 and n3 == 0:
                        continue
                    h_list.append([n1, n2, n3])
        h = torch.tensor(h_list, dtype=torch.float32)
        self.register_buffer("h", h)

        k_min = 1e-3
        k_max = 5.0
        centers_list = []
        for _ in range(n_layers):
            centers = torch.logspace(math.log10(k_min), math.log10(k_max), n_gaussians)
            centers_list.append(centers)
        centers_init = torch.stack(centers_list, dim=0)
        dk = math.log10(k_max / k_min) / max(n_gaussians - 1, 1)
        sigma_init = torch.ones(n_layers, n_gaussians) * (10 ** dk - 1.0)
        log_sigma_init = torch.log(sigma_init)
        weight_init = torch.zeros(n_layers, n_gaussians)

        self.centers = nn.Parameter(centers_init)
        self.log_sigma = nn.Parameter(log_sigma_init)
        self.weights = nn.Parameter(weight_init)

    def forward(self, u: torch.Tensor, pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        H = box
        H_inv = torch.inverse(H)
        G = 2.0 * math.pi * H_inv.transpose(-1, -2)

        k = torch.einsum("ij,bjk->bik", self.h, G)
        k_norm = torch.norm(k, dim=-1)

        phase = torch.einsum("bnc,bkc->bnk", pos, k)
        cos_phase = torch.cos(phase).unsqueeze(1)
        sin_phase = torch.sin(phase).unsqueeze(1)

        u_exp = u.permute(0, 2, 1).unsqueeze(-1)
        rho_real = (u_exp * cos_phase).sum(dim=2)
        rho_imag = (u_exp * (-sin_phase)).sum(dim=2)
        rho2 = rho_real ** 2 + rho_imag ** 2

        k_exp = k_norm.unsqueeze(1).unsqueeze(-1)
        L = self.n_layers
        centers = self.centers.view(1, L, 1, self.n_gaussians)
        sigma = self.log_sigma.exp().view(1, L, 1, self.n_gaussians)
        w = self.weights.view(1, L, 1, self.n_gaussians)

        gauss = torch.exp(-((k_exp - centers) ** 2) / (2.0 * sigma ** 2))
        K_k = (gauss * w).sum(dim=-1)

        return 0.5 * (rho2 * K_k).sum(dim=(1, 2))


class BindingEnergyTransformer(nn.Module):
    def __init__(
        self,
        n_types: int,
        he_type_id: int,
        d_model=256,
        nhead=8,
        num_layers=4,
        use_long_range: bool = True,
        sog_n_gaussians: int = 8,
        sog_r_switch: float = 4.0,
        sog_r_max: float = 12.0,
        n_rbf: int = 16,
        sr_descriptor_cuts: Sequence[float] = (3.0, 12.0),
        lr_descriptor_cuts: Sequence[float] = (3.0, 12.0),
    ):
        super().__init__()
        self.he_type_id = int(he_type_id)
        self.use_long_range = use_long_range
        self.n_layers = num_layers
        self.n_shells = 2
        self.n_rbf_per_shell = n_rbf
        feat_dim = self.n_shells * self.n_rbf_per_shell

        self.sr_descriptor = MultiShellDescriptor(
            he_type_id=self.he_type_id,
            n_shells=self.n_shells,
            n_rbf_per_shell=self.n_rbf_per_shell,
            r_cuts=sr_descriptor_cuts,
        )

        self.lr_descriptor = MultiShellDescriptor(
            he_type_id=self.he_type_id,
            n_shells=self.n_shells,
            n_rbf_per_shell=self.n_rbf_per_shell,
            r_cuts=lr_descriptor_cuts,
        )

        self.sr_net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        self.latent_net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_layers),
        )

        self.lr_block = KSpaceSOGMultiLayer(
            n_layers=num_layers,
            n_gaussians=sog_n_gaussians,
            nmax=2,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        sd = dict(state_dict)
        if "descriptor.r_cuts" in sd:
            sd.setdefault("sr_descriptor.r_cuts", sd["descriptor.r_cuts"])
            sd.setdefault("sr_descriptor.rbf_centers", sd["descriptor.rbf_centers"])
            sd.setdefault("sr_descriptor.rbf_gamma", sd["descriptor.rbf_gamma"])
            sd.setdefault("lr_descriptor.r_cuts", sd["descriptor.r_cuts"])
            sd.setdefault("lr_descriptor.rbf_centers", sd["descriptor.rbf_centers"])
            sd.setdefault("lr_descriptor.rbf_gamma", sd["descriptor.rbf_gamma"])
            sd.pop("descriptor.r_cuts", None)
            sd.pop("descriptor.rbf_centers", None)
            sd.pop("descriptor.rbf_gamma", None)
        try:
            return super().load_state_dict(sd, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(sd, strict=strict)

    def forward(self, Z: torch.Tensor, pos: torch.Tensor, box: torch.Tensor):
        sr_desc = self.sr_descriptor(Z, pos)
        lr_desc = self.lr_descriptor(Z, pos)

        e_sr_atom = self.sr_net(sr_desc).squeeze(-1)
        E_sr = e_sr_atom.sum(dim=1)

        if self.use_long_range:
            u = self.latent_net(lr_desc)
            E_lr = self.lr_block(u, pos, box)
        else:
            E_lr = torch.zeros_like(E_sr)

        E_total = E_sr + E_lr

        he_pos, he_index = find_he_pos(Z, pos, self.he_type_id)
        dist = torch.norm(he_pos - pos, dim=-1)
        mask = torch.ones_like(dist, dtype=torch.bool)
        mask[torch.arange(Z.size(0), device=pos.device), he_index] = False
        dist_masked = dist.masked_fill(~mask, float("inf"))
        r_min, _ = dist_masked.min(dim=-1)

        E_rep = E_sr
        E_res = E_lr
        return E_total, E_rep, E_res, r_min, E_lr


def loss_energy_force(
    E_pred,
    E_true,
    F_pred,
    F_true,
    w_energy: float = 1.0,
    w_force: float = 0.0005,
):
    loss_E = F.mse_loss(E_pred, E_true)
    loss_F = F.mse_loss(F_pred, F_true)
    return w_energy * loss_E + w_force * loss_F, loss_E, loss_F


def train_model(
    root="dataset",
    train_set="set.000",
    val_set="set.001",
    epochs=10000,
    batch_size=64,
    lr=1e-4,
    device="cpu",
    save_dir="checkpoints",
    pretrained_path: Optional[str] = None,
    use_long_range: bool = True,
    w_force: float = 0.0005,
    sr_descriptor_cuts: Sequence[float] = (3.0, 12.0),
    lr_descriptor_cuts: Sequence[float] = (3.0, 12.0),
):
    systems = ("system_ring1", "system_ring2", "system_ring3", "system_ring4")
    steps_per_epoch = 200
    seed = 2025

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    root = Path(root)
    system_dirs = [str(root / s) for s in systems]

    train_datasets = [SystemSetDataset(sd, train_set) for sd in system_dirs]
    val_datasets = [SystemSetDataset(sd, val_set) for sd in system_dirs]

    he_type_ids = [ds.he_type_id for ds in train_datasets]
    if not all(h == he_type_ids[0] for h in he_type_ids):
        raise RuntimeError("He type id mismatch across systems")
    he_type_id = int(he_type_ids[0])

    type_maps = [ds.type_map for ds in train_datasets]
    if not all(tm == type_maps[0] for tm in type_maps):
        raise RuntimeError("type_map mismatch across systems")
    n_types = len(type_maps[0])

    has_forces = all(ds.has_forces for ds in train_datasets)
    w_force_eff = w_force if has_forces else 0.0

    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        for ds in train_datasets
    ]
    val_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
        for ds in val_datasets
    ]

    model = BindingEnergyTransformer(
        n_types=n_types,
        he_type_id=he_type_id,
        d_model=576,
        nhead=12,
        num_layers=8,
        use_long_range=use_long_range,
        sog_n_gaussians=12,
        sog_r_switch=4.18,
        sog_r_max=12.0,
        n_rbf=28,
        sr_descriptor_cuts=sr_descriptor_cuts,
        lr_descriptor_cuts=lr_descriptor_cuts,
    ).to(device)

    if pretrained_path is not None:
        ckpt_path = Path(pretrained_path)
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)

    CM1_PER_EV = 8065.544

    train_iters = [iter(dl) for dl in train_loaders]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_loss_E = 0.0
        total_loss_F = 0.0
        total_mae_E = 0.0

        for _ in range(steps_per_epoch):
            s = int(rng.integers(0, len(train_iters)))
            try:
                Z, pos, E_true, F_true, _, box = next(train_iters[s])
            except StopIteration:
                train_iters[s] = iter(train_loaders[s])
                Z, pos, E_true, F_true, _, box = next(train_iters[s])

            Z = Z.to(device)
            pos = pos.to(device)
            box = box.to(device)
            E_true = E_true.to(device)
            F_true = F_true.to(device)

            pos.requires_grad_(True)
            optimizer.zero_grad()

            E_pred, _, _, _, _ = model(Z, pos, box)
            grad_pos = torch.autograd.grad(E_pred.sum(), pos, create_graph=True)[0]
            F_pred = -grad_pos

            loss, loss_E, loss_F = loss_energy_force(
                E_pred, E_true, F_pred, F_true,
                w_energy=1.0, w_force=w_force_eff
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_loss_E += float(loss_E.item())
            total_loss_F += float(loss_F.item())
            total_mae_E += float(torch.mean(torch.abs(E_pred - E_true)).item())

        avg_loss = total_loss / steps_per_epoch
        avg_loss_E = total_loss_E / steps_per_epoch
        avg_loss_F = total_loss_F / steps_per_epoch
        avg_mae_E = total_mae_E / steps_per_epoch
        avg_mae_meV = avg_mae_E * 1000.0
        avg_mae_cm1 = avg_mae_E * CM1_PER_EV

        model.eval()
        val_mae_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for vloader in val_loaders:
                for Z, pos, E_true, _, _, box in vloader:
                    Z = Z.to(device)
                    pos = pos.to(device)
                    box = box.to(device)
                    E_true = E_true.to(device)
                    E_pred, _, _, _, _ = model(Z, pos, box)
                    val_mae_sum += float(torch.mean(torch.abs(E_pred - E_true)).item())
                    val_batches += 1

        val_mae_E = val_mae_sum / max(val_batches, 1)
        val_mae_meV = val_mae_E * 1000.0
        val_mae_cm1 = val_mae_E * CM1_PER_EV

        print(
            f"Epoch {epoch:05d} | lr={optimizer.param_groups[0]['lr']:.2e} "
            f"| train_loss={avg_loss:.6e} "
            f"| loss_E={avg_loss_E:.6e}, loss_F={avg_loss_F:.6e} "
            f"| train_MAE_E={avg_mae_E:.6e} eV ({avg_mae_meV:.3f} meV, {avg_mae_cm1:.3f} cm^-1) "
            f"| val_MAE_E={val_mae_E:.6e} eV ({val_mae_meV:.3f} meV, {val_mae_cm1:.3f} cm^-1)"
        )

        if epoch % 2000 == 0:
            ckpt_path = save_dir / f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

        scheduler.step()

    final_path = save_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    return model


if __name__ == "__main__":
    _ = train_model(
        root="dataset",
        train_set="set.000",
        val_set="set.001",
        epochs=10000,
        batch_size=64,
        lr=1e-4,
        device="cpu",
        save_dir="checkpoints",
        pretrained_path=None,
        use_long_range=True,
        w_force=0.005,
        sr_descriptor_cuts=(3.0, 12.0),
        lr_descriptor_cuts=(3.0, 12.0),
    )
