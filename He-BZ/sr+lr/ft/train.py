import math
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_type_map(type_map_path: Path):
    txt = type_map_path.read_text(encoding="utf-8").strip().splitlines()
    return [t.strip() for t in txt if t.strip()]


def load_type_raw(type_raw_path: Path, n_atoms: int):
    return np.fromfile(type_raw_path, dtype=np.int32)


def compute_r_min_numpy(coords: np.ndarray, he_index: int) -> np.ndarray:
    he_pos = coords[:, he_index:he_index + 1, :]
    diff = he_pos - coords
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.ones_like(dist, dtype=bool)
    mask[:, he_index] = False
    dist_masked = np.where(mask, dist, np.inf)
    return dist_masked.min(axis=-1)


class BenzeneHeDataset(Dataset):
    def __init__(self, root="dataset", set_name="set.000"):
        self.root = Path(root)
        self.set_dir = self.root / set_name

        coords = np.load(self.set_dir / "coord.npy")
        boxes = np.load(self.set_dir / "box.npy")
        energies = np.load(self.set_dir / "energy.npy")

        if coords.ndim == 2:
            M, D = coords.shape
            coords = coords.reshape(M, D // 3, 3)

        if boxes.ndim == 2 and boxes.shape[1] == 9:
            boxes = boxes.reshape(-1, 3, 3)
        elif boxes.ndim != 3:
            boxes = boxes.reshape(1, 3, 3)

        if energies.ndim == 0:
            energies = energies[None]

        self.coords = coords.astype(np.float32)
        self.boxes = boxes.astype(np.float32)
        self.energies = energies.astype(np.float32)
        self.M, self.N, _ = self.coords.shape

        self.type_map = load_type_map(self.root / "type_map.raw")
        self.type_raw = load_type_raw(self.set_dir / "type.raw", self.N).astype(np.int64)

        he_type_id = self.type_map.index("He")
        self.he_index = int(np.where(self.type_raw == he_type_id)[0][0])

        self.r_min = compute_r_min_numpy(self.coords, self.he_index).astype(np.float32)

        force_path = self.set_dir / "force.npy"
        if force_path.is_file():
            forces = np.load(force_path)
            if forces.ndim == 2 and forces.shape[1] == 3 * self.N:
                forces = forces.reshape(self.M, self.N, 3)
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


class MultiShellDescriptor(nn.Module):

    def __init__(
        self,
        he_index: int,
        n_shells: int = 2,
        n_rbf_per_shell: int = 16,
        r_cuts: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.he_index = he_index
        self.n_shells = n_shells
        self.n_rbf_per_shell = n_rbf_per_shell

        if r_cuts is None:
            r_cuts = [1.7, 8.0]
        if len(r_cuts) != n_shells:
            raise ValueError(f"len(r_cuts)={len(r_cuts)} must equal n_shells={n_shells}")

        self.register_buffer("r_cuts", torch.tensor(r_cuts, dtype=torch.float32))
        self._build_rbf_buffers(r_cuts)

    def _build_rbf_buffers(self, r_cuts: Sequence[float]):
        centers_list = []
        gamma_list = []

        for r_cut in r_cuts:
            r_cut = float(r_cut)
            centers = torch.linspace(0.0, r_cut, self.n_rbf_per_shell)
            dr = r_cut / max(self.n_rbf_per_shell, 1)
            gamma = 1.0 / (2.0 * dr * dr)
            centers_list.append(centers)
            gamma_list.append(gamma)

        self.register_buffer("rbf_centers", torch.stack(centers_list, dim=0))
        self.register_buffer("rbf_gamma", torch.tensor(gamma_list, dtype=torch.float32))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        he_pos = pos[:, self.he_index:self.he_index + 1, :]
        diff = pos - he_pos
        r = torch.norm(diff, dim=-1)

        feats = []

        for s in range(self.n_shells):
            r_cut = self.r_cuts[s]
            centers = self.rbf_centers[s].view(1, 1, -1)
            gamma = self.rbf_gamma[s]

            fc = 0.5 * (torch.cos(torch.clamp(r, max=r_cut) * math.pi / r_cut) + 1.0)
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
                    if n1 != 0 or n2 != 0 or n3 != 0:
                        h_list.append([n1, n2, n3])

        self.register_buffer("h", torch.tensor(h_list, dtype=torch.float32))
        self.n_h = len(h_list)

        k_min = 1e-3
        k_max = 5.0

        centers_list = [
            torch.logspace(math.log10(k_min), math.log10(k_max), n_gaussians)
            for _ in range(n_layers)
        ]

        centers_init = torch.stack(centers_list, dim=0)

        dk = math.log10(k_max / k_min) / max(n_gaussians - 1, 1)
        sigma_init = torch.ones(n_layers, n_gaussians) * (10 ** dk - 1.0)

        self.centers = nn.Parameter(centers_init)
        self.log_sigma = nn.Parameter(torch.log(sigma_init))
        self.weights = nn.Parameter(torch.zeros(n_layers, n_gaussians))

    def forward(self, u: torch.Tensor, pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        L = self.n_layers

        H_inv = torch.inverse(box)
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
        centers = self.centers.view(1, L, 1, self.n_gaussians)
        sigma = self.log_sigma.exp().view(1, L, 1, self.n_gaussians)
        w = self.weights.view(1, L, 1, self.n_gaussians)

        gauss = torch.exp(-((k_exp - centers) ** 2) / (2.0 * sigma ** 2))
        K_k = (gauss * w).sum(dim=-1)

        return 0.5 * (rho2 * K_k).sum(dim=(1, 2))


class BindingEnergyTransformer(nn.Module):
    def __init__(
        self,
        n_types,
        he_index: int,
        d_model=256,
        nhead=8,
        num_layers=4,
        use_long_range: bool = True,
        sog_n_gaussians: int = 8,
        sog_r_switch: float = 4.0,
        sog_r_max: float = 12.0,
        n_rbf: int = 16,
        sr_descriptor_cuts: Sequence[float] = (1.7, 8.0),
        lr_descriptor_cuts: Sequence[float] = (1.7, 8.0),
        preserve_descriptor_cuts_on_load: bool = True,
    ):

        super().__init__()

        self.he_index = he_index
        self.n_layers = num_layers
        self.n_shells = 2
        self.n_rbf_per_shell = n_rbf
        self.use_long_range = use_long_range
        self.preserve_descriptor_cuts_on_load = preserve_descriptor_cuts_on_load

        feat_dim = self.n_shells * self.n_rbf_per_shell

        self.sr_descriptor = MultiShellDescriptor(
            he_index=he_index,
            n_shells=self.n_shells,
            n_rbf_per_shell=self.n_rbf_per_shell,
            r_cuts=sr_descriptor_cuts,
        )

        self.lr_descriptor = MultiShellDescriptor(
            he_index=he_index,
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

    def _convert_legacy_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sd = dict(state_dict)

        if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
            sd = dict(sd["model_state_dict"])

        legacy_map = {
            "descriptor.r_cuts": ["sr_descriptor.r_cuts", "lr_descriptor.r_cuts"],
            "descriptor.rbf_centers": ["sr_descriptor.rbf_centers", "lr_descriptor.rbf_centers"],
            "descriptor.rbf_gamma": ["sr_descriptor.rbf_gamma", "lr_descriptor.rbf_gamma"],
        }

        for old_key, new_keys in legacy_map.items():
            if old_key in sd:
                for new_key in new_keys:
                    sd.setdefault(new_key, sd[old_key].clone() if torch.is_tensor(sd[old_key]) else sd[old_key])

        for old_key in legacy_map:
            sd.pop(old_key, None)

        if self.preserve_descriptor_cuts_on_load:
            current_sd = super().state_dict()
            for key in (
                "sr_descriptor.r_cuts",
                "sr_descriptor.rbf_centers",
                "sr_descriptor.rbf_gamma",
                "lr_descriptor.r_cuts",
                "lr_descriptor.rbf_centers",
                "lr_descriptor.rbf_gamma",
            ):
                sd[key] = current_sd[key]

        return sd

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        state_dict = self._convert_legacy_state_dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def forward(self, Z, pos, box):
        sr_desc = self.sr_descriptor(pos)
        lr_desc = self.lr_descriptor(pos)

        e_sr_atom = self.sr_net(sr_desc).squeeze(-1)
        E_sr = e_sr_atom.sum(dim=1)

        if self.use_long_range:
            u = self.latent_net(lr_desc)
            E_lr = self.lr_block(u, pos, box)
        else:
            E_lr = torch.zeros_like(E_sr)

        E_total = E_sr + E_lr

        he_pos = pos[:, self.he_index:self.he_index + 1, :]
        diff = he_pos - pos
        dist = torch.norm(diff, dim=-1)

        mask = torch.ones_like(dist, dtype=torch.bool)
        mask[:, self.he_index] = False

        dist_masked = dist.masked_fill(~mask, float("inf"))
        r_min, _ = dist_masked.min(dim=-1)

        return E_total, E_sr, E_lr, r_min, E_lr


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
    epochs=20000,
    batch_size=64,
    lr=5e-5,
    device="cpu",
    save_dir="checkpoints",
    pretrained_path: Optional[str] = None,
    use_long_range: bool = True,
    w_force: float = 0.0005,
    sr_descriptor_cuts: Sequence[float] = (1.7, 8.0),
    lr_descriptor_cuts: Sequence[float] = (1.7, 8.0),
    preserve_descriptor_cuts_on_load: bool = True,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    train_dataset = BenzeneHeDataset(root, train_set)
    val_dataset = BenzeneHeDataset(root, val_set)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BindingEnergyTransformer(
        n_types=len(train_dataset.type_map),
        he_index=train_dataset.he_index,
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
        preserve_descriptor_cuts_on_load=preserve_descriptor_cuts_on_load,
    ).to(device)

    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location=device)

        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded pretrained model:", pretrained_path)
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)

    CM1_PER_EV = 8065.544
    w_force_eff = w_force if train_dataset.has_forces else 0.0

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_loss_E = 0.0
        total_loss_F = 0.0
        total_mae_E = 0.0
        steps = 0

        for Z, pos, E_true, F_true, r_min_np, box in train_loader:
            Z = Z.to(device)
            pos = pos.to(device)
            box = box.to(device)
            E_true = E_true.to(device)
            F_true = F_true.to(device)

            pos.requires_grad_(True)

            optimizer.zero_grad()

            E_pred, E_rep, E_res, r_min, E_lr = model(Z, pos, box)

            grad_pos = torch.autograd.grad(E_pred.sum(), pos, create_graph=True)[0]
            F_pred = -grad_pos

            loss, loss_E, loss_F = loss_energy_force(
                E_pred,
                E_true,
                F_pred,
                F_true,
                w_energy=1.0,
                w_force=w_force_eff,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_loss_E += loss_E.item()
            total_loss_F += loss_F.item()
            total_mae_E += torch.mean(torch.abs(E_pred - E_true)).item()
            steps += 1

        avg_loss = total_loss / steps
        avg_loss_E = total_loss_E / steps
        avg_loss_F = total_loss_F / steps
        avg_mae_E = total_mae_E / steps
        avg_mae_meV = avg_mae_E * 1000.0
        avg_mae_cm1 = avg_mae_E * CM1_PER_EV

        model.eval()

        val_mae_E = 0.0
        vsteps = 0

        with torch.no_grad():
            for Z, pos, E_true, F_true, r_min_np, box in val_loader:
                Z = Z.to(device)
                pos = pos.to(device)
                box = box.to(device)
                E_true = E_true.to(device)

                E_pred, _, _, _, _ = model(Z, pos, box)

                val_mae_E += torch.mean(torch.abs(E_pred - E_true)).item()
                vsteps += 1

        val_mae_E /= vsteps
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
            torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch}.pt")

        scheduler.step()

    torch.save(model.state_dict(), save_dir / "model_final.pt")

    return model


if __name__ == "__main__":
    train_model(
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

        sr_descriptor_cuts=(1.7, 8.0),
        lr_descriptor_cuts=(1.7, 8.0),
        preserve_descriptor_cuts_on_load=True,
    )
