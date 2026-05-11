"""Core modules for training a Mask-PINN on a 2D Helmholtz problem.

The model solves

    u_xx + u_yy + k^2 u = f(x, y),    (x, y) in [-1, 1]^2,
    u = 0,                            (x, y) on the boundary,

where the manufactured exact solution is

    u(x, y) = sin(a1 * pi * x) sin(a2 * pi * y).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pyDOE import lhs
from torch import nn
from torch.optim import lr_scheduler


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for Mask-PINN training."""

    a1: float = 6.0
    a2: float = 6.0
    k: float = 1.0
    lower_bound: tuple[float, float] = (-1.0, -1.0)
    upper_bound: tuple[float, float] = (1.0, 1.0)
    hidden_dim: int = 128
    num_blocks: int = 5
    num_boundary_points: int = 100
    num_collocation_points: int = 10_000
    grid_size: int = 201
    epochs: int = 50_000
    learning_rate: float = 1.0e-4
    scheduler_gamma: float = 0.9
    scheduler_step_size: int = 1_000
    boundary_weight: float = 100.0
    log_every: int = 100
    mask_init: float = 20.0
    dtype: torch.dtype = torch.float32


def get_device(device: str | None = None) -> torch.device:
    """Return the requested device, or CUDA when available."""

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor to a detached CPU NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(value)!r}.")


def gradient(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute d(output)/d(inputs) with autograd."""

    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )[0]


def build_reference_solution(config: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grid coordinates and the manufactured exact solution."""

    x = np.linspace(config.lower_bound[0], config.upper_bound[0], config.grid_size)
    y = np.linspace(config.lower_bound[1], config.upper_bound[1], config.grid_size)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    exact = np.sin(config.a1 * np.pi * x_grid) * np.sin(config.a2 * np.pi * y_grid)
    x_flat = x_grid.reshape(-1, 1)
    y_flat = y_grid.reshape(-1, 1)
    exact_flat = exact.reshape(-1, 1)
    return x_flat, y_flat, exact_flat


class Mask(nn.Module):
    """Learnable element-wise mask m(z) = 1 - exp(-(alpha z)^2)."""

    def __init__(self, width: int, init_value: float = 20.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(init_value * torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = self.alpha * x
        return 1.0 - torch.exp(-(scaled**2))


class MaskBlock(nn.Module):
    """Residual block with masked GELU activations."""

    def __init__(self, width: int, mask_init: float = 20.0) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.mask1 = Mask(width, mask_init)
        self.mask2 = Mask(width, mask_init)
        self.layer1 = nn.Linear(width, width)
        self.layer2 = nn.Linear(width, width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear layers with Xavier-normal weights."""

        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(inputs)
        hidden = self.activation(hidden) * self.mask1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.activation(hidden) * self.mask2(hidden)
        return hidden + inputs


class MaskPINN(nn.Module):
    """Fully connected Mask-PINN network."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_blocks: int = 5,
        mask_init: float = 20.0,
    ) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_mask = Mask(hidden_dim, mask_init)
        self.blocks = nn.Sequential(
            *[MaskBlock(hidden_dim, mask_init) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize input and output layers with Xavier-normal weights."""

        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        features = self.input_layer(coordinates)
        features = self.activation(features) * self.input_mask(features)
        features = self.blocks(features)
        return self.output_layer(features)


class PhysicsInformedNN:
    """Trainer for a 2D Helmholtz Mask-PINN."""

    def __init__(
        self,
        config: TrainConfig,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        exact_eval: np.ndarray,
        output_dir: str | Path,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or get_device()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lower_bound = np.asarray(config.lower_bound, dtype=np.float64)
        self.upper_bound = np.asarray(config.upper_bound, dtype=np.float64)
        self.exact_eval = exact_eval

        self.x_eval = self._tensor(x_eval)
        self.y_eval = self._tensor(y_eval)

        self.model = MaskPINN(
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            mask_init=config.mask_init,
        ).to(device=self.device, dtype=config.dtype)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
        )
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=config.scheduler_gamma,
        )

        self.iteration = 0
        self.last_iteration_time = 0.0
        self.history: list[tuple[int, float, float]] = []

    def _tensor(self, array: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(
            array,
            dtype=self.config.dtype,
            device=self.device,
            requires_grad=requires_grad,
        )

    def solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the neural approximation u_theta(x, y)."""

        return self.model(torch.cat([x, y], dim=1))

    def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Helmholtz residual."""

        cfg = self.config
        forcing = (
            -((cfg.a1 * np.pi) ** 2) * torch.sin(cfg.a1 * np.pi * x) * torch.sin(cfg.a2 * np.pi * y)
            -((cfg.a2 * np.pi) ** 2) * torch.sin(cfg.a1 * np.pi * x) * torch.sin(cfg.a2 * np.pi * y)
            +(cfg.k**2) * torch.sin(cfg.a1 * np.pi * x) * torch.sin(cfg.a2 * np.pi * y)
        )

        u = self.solution(x, y)
        u_x = gradient(u, x)
        u_y = gradient(u, y)
        u_xx = gradient(u_x, x)
        u_yy = gradient(u_y, y)
        return u_xx + u_yy + (cfg.k**2) * u - forcing

    def sample_training_points(self) -> tuple[torch.Tensor, ...]:
        """Sample boundary and collocation points using Latin hypercube sampling."""

        cfg = self.config
        x_boundary = self.lower_bound[0] + (self.upper_bound[0] - self.lower_bound[0]) * lhs(1, cfg.num_boundary_points)
        y_boundary = self.lower_bound[1] + (self.upper_bound[1] - self.lower_bound[1]) * lhs(1, cfg.num_boundary_points)

        x_left = self._tensor(np.full_like(y_boundary, self.lower_bound[0]), requires_grad=True)
        y_left = self._tensor(y_boundary, requires_grad=True)
        x_right = self._tensor(np.full_like(y_boundary, self.upper_bound[0]), requires_grad=True)
        y_right = self._tensor(y_boundary, requires_grad=True)
        x_bottom = self._tensor(x_boundary, requires_grad=True)
        y_bottom = self._tensor(np.full_like(x_boundary, self.lower_bound[1]), requires_grad=True)
        x_top = self._tensor(x_boundary, requires_grad=True)
        y_top = self._tensor(np.full_like(x_boundary, self.upper_bound[1]), requires_grad=True)

        collocation = self.lower_bound + (self.upper_bound - self.lower_bound) * lhs(2, cfg.num_collocation_points)
        x_collocation = self._tensor(collocation[:, 0:1], requires_grad=True)
        y_collocation = self._tensor(collocation[:, 1:2], requires_grad=True)

        return (
            x_left,
            y_left,
            x_right,
            y_right,
            x_bottom,
            y_bottom,
            x_top,
            y_top,
            x_collocation,
            y_collocation,
        )

    def compute_loss(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total loss, residual loss, and boundary loss."""

        (
            x_left,
            y_left,
            x_right,
            y_right,
            x_bottom,
            y_bottom,
            x_top,
            y_top,
            x_collocation,
            y_collocation,
        ) = self.sample_training_points()

        residual_loss = torch.mean(self.residual(x_collocation, y_collocation) ** 2)
        boundary_loss = (
            torch.mean(self.solution(x_left, y_left) ** 2)
            + torch.mean(self.solution(x_right, y_right) ** 2)
            + torch.mean(self.solution(x_bottom, y_bottom) ** 2)
            + torch.mean(self.solution(x_top, y_top) ** 2)
        )
        total_loss = residual_loss + self.config.boundary_weight * boundary_loss
        return total_loss, residual_loss, boundary_loss

    @torch.no_grad()
    def relative_l2_error(self) -> float:
        """Compute relative L2 error on the evaluation grid."""

        prediction = to_numpy(self.solution(self.x_eval, self.y_eval))
        return float(np.linalg.norm(self.exact_eval - prediction, 2) / np.linalg.norm(self.exact_eval, 2))

    def train_step(self) -> tuple[float, float, float]:
        """Run one optimization step."""

        self.optimizer.zero_grad(set_to_none=True)
        total_loss, residual_loss, boundary_loss = self.compute_loss()
        total_loss.backward()
        self.optimizer.step()
        self.iteration += 1
        return float(total_loss.item()), float(residual_loss.item()), float(boundary_loss.item())

    def train(self) -> np.ndarray:
        """Train the model and return logged history as [iteration, rel_l2, loss]."""

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {trainable_params:,}")

        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            loss, residual_loss, boundary_loss = self.train_step()
            self.last_iteration_time = time.time() - start_time

            if epoch % self.config.scheduler_step_size == 0:
                self.scheduler.step()

            if self.iteration % self.config.log_every == 0:
                rel_l2 = self.relative_l2_error()
                self.history.append((self.iteration, rel_l2, loss))
                print(
                    f"Iter {self.iteration:6d} | "
                    f"loss {loss:.3e} | residual {residual_loss:.3e} | "
                    f"boundary {boundary_loss:.3e} | rel_L2 {rel_l2:.3e} | "
                    f"time/iter {self.last_iteration_time:.2e}s"
                )
                self.save_history()

        return self.save_history()

    def save_history(self) -> np.ndarray:
        """Save training history to disk."""

        history = np.asarray(self.history, dtype=np.float64)
        path = self.output_dir / "losses.txt"
        if history.size == 0:
            history = np.empty((0, 3), dtype=np.float64)
        np.savetxt(path, history, fmt="%.10f %.10f %.10f", header="iter rel_l2 loss")
        return history

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimizer, scheduler, and config state."""

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "iteration": self.iteration,
                "config": self.config,
            },
            path,
        )

    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predict u(x, y) for NumPy coordinate arrays."""

        self.model.eval()
        with torch.no_grad():
            x_tensor = self._tensor(x)
            y_tensor = self._tensor(y)
            return to_numpy(self.solution(x_tensor, y_tensor))
