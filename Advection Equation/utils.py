"""Model and training utilities for a 1D linear advection PINN.

Equation:
    u_t + beta * u_x = 0,  x in [0, 2*pi], t in [0, 1]

Exact solution used in the default experiment:
    u(x, t) = sin(x - beta*t)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from pyDOE import lhs
from torch import nn
from torch.optim import lr_scheduler


Array = np.ndarray
Tensor = torch.Tensor


@dataclass(frozen=True)
class AdvectionConfig:
    """Configuration for the advection PINN experiment."""

    beta: float = 30.0
    x_min: float = 0.0
    x_max: float = 2.0 * np.pi
    t_min: float = 0.0
    t_max: float = 1.0
    n_initial: int = 200
    n_boundary: int = 200
    n_collocation: int = 5000
    hidden_dim: int = 256
    num_blocks: int = 6
    learning_rate: float = 1.0e-3
    adam_betas: Tuple[float, float] = (0.99, 0.999)
    scheduler_gamma: float = 0.9
    scheduler_step: int = 1000
    epochs: int = 50_000
    print_every: int = 500
    checkpoint_every: int = 5000
    mask_init: float = 1.0
    boundary_weight: float = 1.0
    initial_weight: float = 1.0
    residual_weight: float = 1.0


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(value: Tensor | Array) -> Array:
    """Convert a torch tensor to a NumPy array without keeping gradients."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(value)!r}")


def gradient(output: Tensor, input_: Tensor) -> Tensor:
    """Compute d(output) / d(input_) using automatic differentiation."""

    return torch.autograd.grad(
        output,
        input_,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )[0]


class MaskLayer(nn.Module):
    """Element-wise learnable mask m(z) = 1 - exp(-(a z)^2)."""

    def __init__(self, width: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        h = self.scale * x
        return 1.0 - torch.exp(-(h**2))


class MaskBlock(nn.Module):
    """Residual block with masked nonlinear transformations."""

    def __init__(self, width: int, mask_init: float = 1.0) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.mask1 = MaskLayer(width, mask_init)
        self.mask2 = MaskLayer(width, mask_init)
        self.layer1 = nn.Linear(width, width)
        self.layer2 = nn.Linear(width, width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        z1 = self.layer1(inputs)
        h1 = self.activation(z1)
        z2 = self.layer2(h1 * self.mask1(z1))
        h2 = self.activation(z2)
        return h2 * self.mask2(z2) + inputs


class MaskPINN(nn.Module):
    """Fully-connected PINN with masked residual blocks."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_blocks: int = 6,
        mask_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_mask = MaskLayer(hidden_dim, mask_init)
        self.blocks = nn.Sequential(
            *[MaskBlock(hidden_dim, mask_init) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        z = self.input_layer(inputs)
        h = self.activation(z) * self.input_mask(z)
        h = self.blocks(h)
        return self.output_layer(h)


class AdvectionPINN:
    """Trainer wrapper for the 1D advection PINN."""

    def __init__(
        self,
        x_exact: Array,
        config: AdvectionConfig,
        output_dir: str | Path,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or get_device()
        self.dtype = dtype
        self.iteration = 0
        self.last_iter_time = 0.0
        self.loss: Tensor | None = None
        self.history: list[tuple[int, float, float]] = []

        self.x_test = torch.tensor(
            x_exact[:, 0:1], requires_grad=True, dtype=dtype, device=self.device
        )
        self.t_test = torch.tensor(
            x_exact[:, 1:2], requires_grad=True, dtype=dtype, device=self.device
        )
        self.exact = x_exact[:, 2:3]

        self.model = MaskPINN(
            input_dim=2,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_blocks=config.num_blocks,
            mask_init=config.mask_init,
        ).to(device=self.device, dtype=dtype)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
        )
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.scheduler_gamma
        )

    def net_u(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(torch.cat((x, t), dim=1))

    def net_residual(self, x: Tensor, t: Tensor) -> Tensor:
        u = self.net_u(x, t)
        u_t = gradient(u, t)
        u_x = gradient(u, x)
        return u_t + self.config.beta * u_x

    def sample_training_points(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config

        x0 = cfg.x_min + (cfg.x_max - cfg.x_min) * lhs(1, cfg.n_initial)
        t0 = np.zeros_like(x0) + cfg.t_min

        tb = cfg.t_min + (cfg.t_max - cfg.t_min) * lhs(1, cfg.n_boundary)
        x_left = np.zeros_like(tb) + cfg.x_min
        x_right = np.zeros_like(tb) + cfg.x_max

        xf = cfg.x_min + (cfg.x_max - cfg.x_min) * lhs(1, cfg.n_collocation)
        tf = cfg.t_min + (cfg.t_max - cfg.t_min) * lhs(1, cfg.n_collocation)

        def tensor(array: Array) -> Tensor:
            return torch.tensor(
                array, requires_grad=True, dtype=self.dtype, device=self.device
            )

        return (
            tensor(x0),
            tensor(t0),
            tensor(x_left),
            tensor(tb),
            tensor(x_right),
            tensor(tb),
            tensor(xf),
            tensor(tf),
        )

    def training_step(self) -> None:
        cfg = self.config
        self.optimizer.zero_grad()

        x0, t0, x_left, t_left, x_right, t_right, xf, tf = self.sample_training_points()

        u0 = self.net_u(x0, t0)
        u_left = self.net_u(x_left, t_left)
        u_right = self.net_u(x_right, t_right)
        residual = self.net_residual(xf, tf)

        loss_initial = torch.mean((u0 - torch.sin(x0)) ** 2)
        loss_boundary = torch.mean((u_left - u_right) ** 2)
        loss_residual = torch.mean(residual**2)

        self.loss = (
            cfg.initial_weight * loss_initial
            + cfg.boundary_weight * loss_boundary
            + cfg.residual_weight * loss_residual
        )
        self.loss.backward()
        self.optimizer.step()
        self.iteration += 1

    def relative_l2_error(self) -> float:
        self.model.eval()
        with torch.no_grad():
            prediction = self.net_u(self.x_test, self.t_test)
        self.model.train()
        pred_np = to_numpy(prediction)
        return float(np.linalg.norm(self.exact - pred_np, 2) / np.linalg.norm(self.exact, 2))

    def save_history(self) -> None:
        history = np.array(self.history, dtype=float)
        if history.size == 0:
            return
        np.savetxt(
            self.output_dir / "losses.txt",
            history,
            fmt="%.10f %.10f %.10f",
            header="iter rel_l2 loss",
            comments="",
        )

    def save_checkpoint(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "iteration": self.iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": None if self.loss is None else self.loss.detach().cpu(),
                "config": self.config,
            },
            self.checkpoint_dir / f"checkpoint_{epoch:06d}.pt",
        )

    def train(self) -> Array:
        parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {parameter_count}")
        print(f"Device: {self.device}")

        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            start = time.time()
            self.training_step()
            self.last_iter_time = time.time() - start

            if epoch % self.config.scheduler_step == 0:
                self.scheduler.step()

            if epoch % self.config.print_every == 0:
                rel_l2 = self.relative_l2_error()
                loss_value = float(self.loss.detach().cpu()) if self.loss is not None else np.nan
                self.history.append((self.iteration, rel_l2, loss_value))
                self.save_history()
                print(
                    f"Iter {self.iteration:6d} | "
                    f"Loss {loss_value:.3e} | "
                    f"Rel_L2 {rel_l2:.3e} | "
                    f"time/iter {self.last_iter_time:.2e}s"
                )

            if self.config.checkpoint_every > 0 and epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch)

        self.save_history()
        return np.array(self.history, dtype=float)

    def predict(self, x: Array, t: Array) -> Array:
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=self.dtype, device=self.device)
        t_tensor = torch.tensor(t, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            u = self.net_u(x_tensor, t_tensor)
        return to_numpy(u)
