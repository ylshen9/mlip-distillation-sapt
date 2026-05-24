import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.special import sph_harm_y as _sph
except Exception:
    from scipy.special import sph_harm as _sph


LAM_VAR_START = 1.0
LAM_VAR_END = 0.5
LAM_CENTER_START = 0.2
LAM_CENTER_END = 0.05
ANNEAL_START_FRAC = 0.30
ANNEAL_END_FRAC = 0.80


def anneal_linear(epoch: int, total_epochs: int, start_frac: float, end_frac: float, v0: float, v1: float) -> float:
    if total_epochs <= 1:
        return float(v1)
    t0 = int(round(start_frac * total_epochs))
    t1 = int(round(end_frac * total_epochs))
    if t1 <= t0:
        return float(v1 if epoch >= t0 else v0)
    if epoch <= t0:
        return float(v0)
    if epoch >= t1:
        return float(v1)
    a = (epoch - t0) / float(t1 - t0)
    return float(v0 + a * (v1 - v0))


def _sph_eval(m, l, phi, theta):
    if getattr(_sph, "__name__", "") == "sph_harm_y":
        return _sph(m, l, theta, phi)
    return _sph(m, l, phi, theta)


def load_type_map(path: Path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_type_raw(path: Path, n_atoms: int):
    arr = np.fromfile(path, dtype=np.int32)
    assert arr.size == n_atoms
    return arr


def compute_r_min_numpy(coords: np.ndarray, he_index: int) -> np.ndarray:
    he = coords[:, he_index:he_index + 1, :]
    d = np.linalg.norm(coords - he, axis=-1)
    d[:, he_index] = np.inf
    return d.min(axis=1)


def _sph_real_design(theta, phi, lmax):
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)
    cols = []
    for l in range(lmax + 1):
        Y0 = _sph_eval(0, l, phi, theta)
        cols.append(np.real(Y0))
        for m in range(1, l + 1):
            Y = _sph_eval(m, l, phi, theta)
            cols.append(np.sqrt(2.0) * ((-1) ** m) * np.real(Y))
            cols.append(np.sqrt(2.0) * ((-1) ** m) * np.imag(Y))
    return np.column_stack(cols)


def load_rc_model_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    lmax_arr = d["lmax"]
    lmax = int(lmax_arr) if np.ndim(lmax_arr) == 0 else int(lmax_arr.item())
    return {"lmax": lmax, "coeff": np.asarray(d["coeff"], dtype=float)}


def rc_center_from_model(theta, phi, model):
    A = _sph_real_design(theta, phi, model["lmax"])
    return A @ model["coeff"]


def cart2sph(u):
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return theta, phi


def compute_rc_center_for_frame(pos: np.ndarray, he_index: int, c_idx: np.ndarray, rc_model: dict, rc_clip=(0.2, 20.0)) -> float:
    c = pos[c_idx].mean(axis=0)
    v = pos[he_index] - c
    r = np.linalg.norm(v)
    u = np.array([0.0, 0.0, 1.0], dtype=float) if r < 1e-8 else (v / r)
    theta, phi = cart2sph(u[None, :])
    Rc = float(rc_center_from_model(theta, phi, rc_model)[0])
    Rc = float(np.clip(Rc, rc_clip[0], rc_clip[1]))
    return Rc


class BenzeneHeDataset(Dataset):
    def __init__(self, root="dataset", set_name="set.000", model_npz_name="rc_fit.npz"):
        self.root = Path(root)
        self.set_dir = self.root / set_name

        coords = np.load(self.set_dir / "coord.npy")
        boxes = np.load(self.set_dir / "box.npy")
        energies = np.load(self.set_dir / "energy.npy")

        if coords.ndim == 3:
            self.M, self.N, _ = coords.shape
            self.coords = coords.astype(np.float32)
        elif coords.ndim == 2:
            self.M, D = coords.shape
            assert D % 3 == 0
            self.N = D // 3
            self.coords = coords.reshape(self.M, self.N, 3).astype(np.float32)
        else:
            raise ValueError("coord.npy must be (M,N,3) or (M,3N)")

        if boxes.ndim == 3:
            self.boxes = boxes.astype(np.float32)
        elif boxes.ndim == 2 and boxes.shape[1] == 9:
            self.boxes = boxes.reshape(-1, 3, 3).astype(np.float32)
        else:
            self.boxes = boxes.reshape(1, 3, 3).astype(np.float32)

        if energies.ndim == 0:
            energies = energies[None]
        assert energies.ndim == 1
        self.energies = energies.astype(np.float32)

        self.type_map = load_type_map(self.root / "type_map.raw")
        self.type_raw = load_type_raw(self.set_dir / "type.raw", self.N).astype(np.int64)

        assert "He" in self.type_map
        he_tid = self.type_map.index("He")
        he_idx = np.where(self.type_raw == he_tid)[0]
        assert he_idx.size == 1
        self.he_index = int(he_idx[0])

        self.r_min = compute_r_min_numpy(self.coords, self.he_index).astype(np.float32)

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

        model_path = (Path(__file__).resolve().parent / model_npz_name).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"rc model not found: {model_path}")
        self.rc_model = load_rc_model_npz(model_path)

        assert "C" in self.type_map
        c_tid = self.type_map.index("C")
        self.c_indices = np.where(self.type_raw == c_tid)[0]
        assert self.c_indices.size >= 6

        self.Rc_center = np.zeros((self.M,), dtype=np.float32)
        for m in range(self.M):
            self.Rc_center[m] = compute_rc_center_for_frame(
                pos=self.coords[m],
                he_index=self.he_index,
                c_idx=self.c_indices,
                rc_model=self.rc_model,
                rc_clip=(0.2, 20.0),
            )

    def __len__(self):
        return self.M

    def __getitem__(self, i):
        Z = torch.from_numpy(self.type_raw)
        pos = torch.from_numpy(self.coords[i])
        E = torch.tensor(self.energies[i], dtype=torch.float32)
        r_min = torch.tensor(self.r_min[i], dtype=torch.float32)
        if self.has_forces:
            Fv = torch.from_numpy(self.forces[i])
        else:
            Fv = torch.zeros(self.N, 3, dtype=torch.float32)
        box = torch.from_numpy(self.boxes[0] if self.boxes.shape[0] == 1 else self.boxes[i])
        Rc = torch.tensor(self.Rc_center[i], dtype=torch.float32)
        return Z, pos, E, Fv, r_min, box, Rc


class MultiShellDescriptorDynamic(nn.Module):
    def __init__(self, he_index: int, n_rbf: int = 28, eps: float = 1e-6, scales: Tuple[float, ...] = (0.3, 1.0)):
        super().__init__()
        self.he_index = he_index
        self.n_rbf = n_rbf
        self.eps = eps
        self.scales = tuple(float(x) for x in scales)

        centers = torch.linspace(0.0, 1.0, n_rbf)
        ds = 1.0 / max(n_rbf - 1, 1)
        gamma = 1.0 / (2.0 * ds * ds)
        self.register_buffer("centers", centers.view(1, 1, -1))
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def forward(self, pos: torch.Tensor, rc_atom: torch.Tensor) -> torch.Tensor:
        he = pos[:, self.he_index:self.he_index + 1, :]
        r = torch.norm(pos - he, dim=-1)
        rc = torch.clamp(rc_atom, min=self.eps)

        feats = []
        for s in self.scales:
            x = (r / (rc * s)).unsqueeze(-1)
            fc = (x < 1.0).float()
            feats.append(torch.exp(-self.gamma * (x - self.centers) ** 2) * fc)
        return torch.cat(feats, dim=-1)


class RcPredictorPair(nn.Module):
    def __init__(self, n_types: int, d_embed: int = 32, hidden: int = 128, rc_min: float = 0.8, rc_max: float = 12.0):
        super().__init__()
        self.embed = nn.Embedding(n_types, d_embed)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed + 3 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.rc_min = float(rc_min)
        self.rc_max = float(rc_max)

    def forward(self, Z: torch.Tensor, pos: torch.Tensor, he_index: int) -> torch.Tensor:
        e = self.embed(Z)
        he = pos[:, he_index:he_index + 1, :]
        rvec = pos - he
        r = torch.norm(rvec, dim=-1, keepdim=True)
        rhat = rvec / (r + 1e-8)
        x = torch.cat([e, rhat, r], dim=-1)
        rc = self.mlp(x).squeeze(-1)
        rc = torch.clamp(rc, self.rc_min, self.rc_max)
        rc = rc.clone()
        rc[:, he_index] = 1.0
        return rc


class KSpaceSOGMultiLayer(nn.Module):
    def __init__(self, n_layers: int = 8, n_gaussians: int = 12, nmax: int = 2):
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

    def forward(self, u_latent: torch.Tensor, pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        H_inv = torch.inverse(box)
        G = 2.0 * math.pi * H_inv.transpose(-1, -2)
        k = torch.einsum("ij,bjk->bik", self.h, G)
        k_norm = torch.norm(k, dim=-1)

        phase = torch.einsum("bnc,bkc->bnk", pos, k)
        cos_phase = torch.cos(phase).unsqueeze(1)
        sin_phase = torch.sin(phase).unsqueeze(1)

        u_exp = u_latent.permute(0, 2, 1).unsqueeze(-1)
        rho_real = (u_exp * cos_phase).sum(dim=2)
        rho_imag = (u_exp * (-sin_phase)).sum(dim=2)
        rho2 = rho_real ** 2 + rho_imag ** 2

        k_exp = k_norm.unsqueeze(1).unsqueeze(-1)
        centers = self.centers.view(1, self.n_layers, 1, self.n_gaussians)
        sigma = self.log_sigma.exp().view(1, self.n_layers, 1, self.n_gaussians)
        w = self.weights.view(1, self.n_layers, 1, self.n_gaussians)
        gauss = torch.exp(-((k_exp - centers) ** 2) / (2.0 * sigma ** 2))
        K_k = (gauss * w).sum(dim=-1)

        E_lr = 0.5 * (rho2 * K_k).sum(dim=(1, 2))
        return E_lr


def variance_penalty(rc_pred: torch.Tensor, type_raw: torch.Tensor, type_map: list, he_index: int) -> torch.Tensor:
    device = rc_pred.device
    type_raw = type_raw.to(device)
    loss = rc_pred.new_zeros(())
    for sym in ("C", "H"):
        if sym not in type_map:
            continue
        tid = type_map.index(sym)
        mask = (type_raw[None, :] == tid).float()
        mask[:, he_index] = 0.0
        cnt = mask.sum(dim=1).clamp_min(1.0)
        mean = (rc_pred * mask).sum(dim=1) / cnt
        var = ((rc_pred - mean[:, None]) ** 2 * mask).sum(dim=1) / cnt
        loss = loss + var.mean()
    return loss


def center_align_loss(rc_pred: torch.Tensor, type_raw: torch.Tensor, type_map: list, he_index: int, Rc_center: torch.Tensor) -> torch.Tensor:
    device = rc_pred.device
    type_raw = type_raw.to(device)
    loss = rc_pred.new_zeros(())
    for sym in ("C", "H"):
        if sym not in type_map:
            continue
        tid = type_map.index(sym)
        mask = (type_raw[None, :] == tid).float()
        mask[:, he_index] = 0.0
        cnt = mask.sum(dim=1).clamp_min(1.0)
        mean_rc = (rc_pred * mask).sum(dim=1) / cnt
        loss = loss + F.mse_loss(mean_rc, Rc_center)
    return loss


class BindingEnergyTransformer(nn.Module):
    def __init__(
        self,
        n_types: int,
        he_index: int,
        d_model: int = 576,
        num_layers: int = 8,
        sog_n_gaussians: int = 12,
        n_rbf: int = 28,
        rc_min: float = 0.8,
        rc_max: float = 12.0,
        rc_lr: float = 9.0,
    ):
        super().__init__()
        self.he_index = he_index
        self.rc_lr = float(rc_lr)

        self.rc_pred_net = RcPredictorPair(n_types=n_types, rc_min=rc_min, rc_max=rc_max)
        self.descriptor = MultiShellDescriptorDynamic(he_index=he_index, n_rbf=n_rbf, scales=(0.3, 1.0))

        feat_dim = 2 * n_rbf

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

        self.lr_block = KSpaceSOGMultiLayer(n_layers=num_layers, n_gaussians=sog_n_gaussians, nmax=2)

    def forward(self, Z: torch.Tensor, pos: torch.Tensor, box: torch.Tensor, Rc_center: torch.Tensor):
        rc_pred = self.rc_pred_net(Z, pos, self.he_index)

        desc_sr = self.descriptor(pos, rc_pred)
        e_sr_atom = self.sr_net(desc_sr).squeeze(-1)
        E_sr = e_sr_atom.sum(dim=1)

        rc_lr_atom = torch.full_like(rc_pred, self.rc_lr)
        rc_lr_atom[:, self.he_index] = 1.0
        desc_lr = self.descriptor(pos, rc_lr_atom)
        u_latent = self.latent_net(desc_lr)
        E_lr = self.lr_block(u_latent, pos, box)

        E_total = E_sr + E_lr

        diff = pos[:, self.he_index:self.he_index + 1, :] - pos
        dist = torch.norm(diff, dim=-1)
        dist[:, self.he_index] = float("inf")
        r_min, _ = dist.min(dim=1)

        return E_total, r_min, rc_pred, E_sr, E_lr


class AutoLossWeights(nn.Module):
    def __init__(self, n_terms: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_terms, dtype=torch.float32))

    def forward(self, losses):
        total = 0.0
        for i, Li in enumerate(losses):
            s = self.log_vars[i]
            total = total + torch.exp(-s) * Li + s
        return total


def loss_energy_force(E_pred, E_true, F_pred, F_true, w_energy: float = 1.0, w_force: float = 0.0005):
    loss_E = F.mse_loss(E_pred, E_true)
    loss_F = F.mse_loss(F_pred, F_true)
    return w_energy * loss_E + w_force * loss_F, loss_E, loss_F


def train_model(
    root="dataset",
    train_set="set.000",
    val_set="set.001",
    model_npz_name="rc_fit.npz",
    epochs=20000,
    batch_size=64,
    lr=5e-5,
    device="cpu",
    save_dir="checkpoints_pair_rc",
    pretrained_path: Optional[str] = None,
    w_force: float = 0.0005,
    rc_lr: float = 9.0,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    train_dataset = BenzeneHeDataset(root, train_set, model_npz_name=model_npz_name)
    val_dataset = BenzeneHeDataset(root, val_set, model_npz_name=model_npz_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BindingEnergyTransformer(
        n_types=len(train_dataset.type_map),
        he_index=train_dataset.he_index,
        d_model=576,
        num_layers=8,
        sog_n_gaussians=12,
        n_rbf=28,
        rc_min=0.8,
        rc_max=12.0,
        rc_lr=rc_lr,
    ).to(device)

    auto_w = AutoLossWeights(n_terms=3).to(device)

    if pretrained_path is not None:
        ckpt_path = Path(pretrained_path)
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(auto_w.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)

    has_forces = train_dataset.has_forces
    w_force_eff = w_force if has_forces else 0.0

    CM1_PER_EV = 8065.544
    type_raw_const = torch.from_numpy(train_dataset.type_raw).long()

    for epoch in range(1, epochs + 1):
        lam_var = anneal_linear(epoch, epochs, ANNEAL_START_FRAC, ANNEAL_END_FRAC, LAM_VAR_START, LAM_VAR_END)
        lam_center = anneal_linear(epoch, epochs, ANNEAL_START_FRAC, ANNEAL_END_FRAC, LAM_CENTER_START, LAM_CENTER_END)

        model.train()
        total_loss = 0.0
        total_mae_E = 0.0
        steps = 0

        for Z, pos, E_true, F_true, r_min_np, box, Rc_center in train_loader:
            Z = Z.to(device)
            pos = pos.to(device)
            box = box.to(device)
            E_true = E_true.to(device)
            F_true = F_true.to(device)
            Rc_center = Rc_center.to(device)

            pos.requires_grad_(True)
            optimizer.zero_grad()

            E_pred, r_min, rc_pred, E_sr, E_lr = model(Z, pos, box, Rc_center)

            grad_pos = torch.autograd.grad(E_pred.sum(), pos, create_graph=True)[0]
            F_pred = -grad_pos

            loss_main, loss_E, loss_F = loss_energy_force(E_pred, E_true, F_pred, F_true, w_energy=1.0, w_force=w_force_eff)

            loss_var = variance_penalty(rc_pred, type_raw_const, train_dataset.type_map, train_dataset.he_index)
            loss_center = center_align_loss(rc_pred, type_raw_const, train_dataset.type_map, train_dataset.he_index, Rc_center)

            loss = auto_w([loss_main, lam_var * loss_var, lam_center * loss_center])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(auto_w.parameters()), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_mae_E += float(torch.mean(torch.abs(E_pred - E_true)).item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        avg_mae_E = total_mae_E / max(steps, 1)
        avg_mae_meV = avg_mae_E * 1000.0
        avg_mae_cm1 = avg_mae_E * CM1_PER_EV

        model.eval()
        val_mae_E = 0.0
        vsteps = 0
        with torch.no_grad():
            for Z, pos, E_true, F_true, r_min_np, box, Rc_center in val_loader:
                Z = Z.to(device)
                pos = pos.to(device)
                box = box.to(device)
                E_true = E_true.to(device)
                Rc_center = Rc_center.to(device)
                E_pred, *_ = model(Z, pos, box, Rc_center)
                val_mae_E += float(torch.mean(torch.abs(E_pred - E_true)).item())
                vsteps += 1
        val_mae_E /= max(vsteps, 1)

        val_mae_meV = val_mae_E * 1000.0
        val_mae_cm1 = val_mae_E * CM1_PER_EV

        lv = auto_w.log_vars.detach().cpu().numpy().tolist()
        print(
            f"Epoch {epoch:05d} | lr={optimizer.param_groups[0]['lr']:.2e} "
            f"| lam_var={lam_var:.4f} lam_center={lam_center:.4f} "
            f"| train_loss={avg_loss:.6e} "
            f"| train_MAE_E={avg_mae_E:.6e} eV ({avg_mae_meV:.3f} meV, {avg_mae_cm1:.3f} cm^-1) "
            f"| val_MAE_E={val_mae_E:.6e} eV ({val_mae_meV:.3f} meV, {val_mae_cm1:.3f} cm^-1) "
            f"| log_vars={lv}"
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
        model_npz_name="rc_fit.npz",
        epochs=10000,
        batch_size=64,
        lr=1e-4,
        device="cpu",
        save_dir="checkpoints",
        pretrained_path=None,
        w_force=0.005,
        rc_lr=12.0,
    )
