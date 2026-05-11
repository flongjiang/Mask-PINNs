"""Core modules for training a Mask-PINN on a 1D linear advection problem.

The model solves

    u_t + beta u_x = 0,      x in [0, 2*pi], t in [0, 1],
    u(x, 0) = sin(x),
    u(0, t) = u(2*pi, t),

where the manufactured exact solution is

    u(x, t) = sin(x - beta t).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pyDOE import lhs
from torch import nn
from torch.optim import lr_scheduler


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for Mask-PINN training."""

    beta: float = 30.0
    x_lower: float = 0.0
    x_upper: float = 2.0 * np.pi
    t_lower: float = 0.0
    t_upper: float = 1.0
    hidden_dim: int = 256
    num_blocks: int = 6
    num_initial_points: int = 200
    num_boundary_points: int = 200
    num_collocation_points: int = 5_000
    grid_size: int = 200
    epochs: int = 50_000
    learning_rate: float = 1.0e-3
    scheduler_gamma: float = 0.9
    scheduler_step_size: int = 1_000
    residual_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 1.0
    log_every: int = 500
    mask_init: float = 1.0
    dtype: torch.dtype = torch.float32


def get_device(device: str | None = None) -> torch.device:
    """Return the requested device, or CUDA when available."""

    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
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

    x = np.linspace(config.x_lower, config.x_upper, config.grid_size)
    t = np.linspace(config.t_lower, config.t_upper, config.grid_size)
    x_grid, t_grid = np.meshgrid(x, t, indexing="xy")

    exact = np.sin(x_grid - config.beta * t_grid)
    x_flat = x_grid.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)
    exact_flat = exact.reshape(-1, 1)
    return x_flat, t_flat, exact_flat


class Mask(nn.Module):
    """Learnable element-wise mask m(z) = 1 - exp(-(alpha z)^2)."""

    def __init__(self, width: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(init_value * torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = self.alpha * x
        return 1.0 - torch.exp(-(scaled**2))


class MaskBlock(nn.Module):
    """Residual block with masked GELU activations."""

    def __init__(self, width: int, mask_init: float = 1.0) -> None:
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
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_blocks: int = 6,
        mask_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_mask = Mask(hidden_dim, mask_init)
        self.blocks = nn.Sequential(*[MaskBlock(hidden_dim, mask_init) for _ in range(num_blocks)])
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
    """Trainer for a 1D linear advection Mask-PINN."""

    def __init__(
        self,
        config: TrainConfig,
        x_eval: np.ndarray,
        t_eval: np.ndarray,
        exact_eval: np.ndarray,
        output_dir: str | Path,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or get_device()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exact_eval = exact_eval
        self.x_eval = self._tensor(x_eval)
        self.t_eval = self._tensor(t_eval)

        self.model = MaskPINN(
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            mask_init=config.mask_init,
        ).to(device=self.device, dtype=config.dtype)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.99, 0.999),
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

    def solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the neural approximation u_theta(x, t)."""

        return self.model(torch.cat([x, t], dim=1))

    def residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the advection residual u_t + beta u_x."""

        u = self.solution(x, t)
        u_t = gradient(u, t)
        u_x = gradient(u, x)
        return u_t + self.config.beta * u_x

    def sample_training_points(self) -> tuple[torch.Tensor, ...]:
        """Sample initial, periodic-boundary, and collocation points using LHS."""

        cfg = self.config

        x_initial = cfg.x_lower + (cfg.x_upper - cfg.x_lower) * lhs(1, cfg.num_initial_points)
        t_initial = np.full_like(x_initial, cfg.t_lower)

        t_boundary = cfg.t_lower + (cfg.t_upper - cfg.t_lower) * lhs(1, cfg.num_boundary_points)
        x_left = np.full_like(t_boundary, cfg.x_lower)
        x_right = np.full_like(t_boundary, cfg.x_upper)

        x_collocation = cfg.x_lower + (cfg.x_upper - cfg.x_lower) * lhs(1, cfg.num_collocation_points)
        t_collocation = cfg.t_lower + (cfg.t_upper - cfg.t_lower) * lhs(1, cfg.num_collocation_points)

        return (
            self._tensor(x_initial, requires_grad=True),
            self._tensor(t_initial, requires_grad=True),
            self._tensor(x_left, requires_grad=True),
            self._tensor(t_boundary, requires_grad=True),
            self._tensor(x_right, requires_grad=True),
            self._tensor(t_boundary, requires_grad=True),
            self._tensor(x_collocation, requires_grad=True),
            self._tensor(t_collocation, requires_grad=True),
        )

    def compute_loss(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total loss, residual loss, initial loss, and boundary loss."""

        (
            x_initial,
            t_initial,
            x_left,
            t_left,
            x_right,
            t_right,
            x_collocation,
            t_collocation,
        ) = self.sample_training_points()

        residual_loss = torch.mean(self.residual(x_collocation, t_collocation) ** 2)
        initial_loss = torch.mean((self.solution(x_initial, t_initial) - torch.sin(x_initial)) ** 2)
        boundary_loss = torch.mean((self.solution(x_left, t_left) - self.solution(x_right, t_right)) ** 2)

        total_loss = (
            self.config.residual_weight * residual_loss
            + self.config.initial_weight * initial_loss
            + self.config.boundary_weight * boundary_loss
        )
        return total_loss, residual_loss, initial_loss, boundary_loss

    @torch.no_grad()
    def relative_l2_error(self) -> float:
        """Compute relative L2 error on the evaluation grid."""

        prediction = to_numpy(self.solution(self.x_eval, self.t_eval))
        return float(np.linalg.norm(self.exact_eval - prediction, 2) / np.linalg.norm(self.exact_eval, 2))

    def train_step(self) -> tuple[float, float, float, float]:
        """Run one optimization step."""

        self.optimizer.zero_grad(set_to_none=True)
        total_loss, residual_loss, initial_loss, boundary_loss = self.compute_loss()
        total_loss.backward()
        self.optimizer.step()
        self.iteration += 1
        return (
            float(total_loss.item()),
            float(residual_loss.item()),
            float(initial_loss.item()),
            float(boundary_loss.item()),
        )

    def train(self) -> np.ndarray:
        """Train the model and return logged history as [iteration, rel_l2, loss]."""

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {trainable_params:,}")

        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            loss, residual_loss, initial_loss, boundary_loss = self.train_step()
            self.last_iteration_time = time.time() - start_time

            if epoch % self.config.scheduler_step_size == 0:
                self.scheduler.step()

            if self.iteration % self.config.log_every == 0:
                rel_l2 = self.relative_l2_error()
                self.history.append((self.iteration, rel_l2, loss))
                print(
                    f"Iter {self.iteration:6d} | "
                    f"loss {loss:.3e} | residual {residual_loss:.3e} | "
                    f"initial {initial_loss:.3e} | boundary {boundary_loss:.3e} | "
                    f"rel_L2 {rel_l2:.3e} | time/iter {self.last_iteration_time:.2e}s"
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

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict u(x, t) for NumPy coordinate arrays."""

        self.model.eval()
        with torch.no_grad():
            x_tensor = self._tensor(x)
            t_tensor = self._tensor(t)
            return to_numpy(self.solution(x_tensor, t_tensor))
