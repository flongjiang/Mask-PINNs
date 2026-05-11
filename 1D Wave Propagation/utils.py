"""Core utilities for training a Mask-PINN on a 1D wave equation."""

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
    c: float = 1.0
    lower_bound_x: float = 0.0
    upper_bound_x: float = 1.0
    lower_bound_t: float = 0.0
    upper_bound_t: float = 5.0
    hidden_dim: int = 64
    num_blocks: int = 3
    num_initial_points: int = 400
    num_boundary_points: int = 400
    num_collocation_points: int = 5000
    grid_size: int = 200
    epochs: int = 20000
    learning_rate: float = 1e-3
    scheduler_gamma: float = 0.9
    scheduler_step_size: int = 500
    log_every: int = 200
    mask_init: float = 1.0
    dtype: torch.dtype = torch.float32


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def gradient(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )[0]


def build_reference_solution(
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    x = np.linspace(config.lower_bound_x, config.upper_bound_x, config.grid_size)
    t = np.linspace(config.lower_bound_t, config.upper_bound_t, config.grid_size)

    x_grid, t_grid = np.meshgrid(x, t)

    exact = np.sin(np.pi * x_grid) * np.cos(config.c * np.pi * t_grid)

    return (
        x_grid.reshape(-1, 1),
        t_grid.reshape(-1, 1),
        exact.reshape(-1, 1),
    )


class Mask(nn.Module):
    def __init__(self, width: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(init_value * torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = self.alpha * x
        return 1.0 - torch.exp(-(scaled**2))


class MaskBlock(nn.Module):
    def __init__(self, width: int, mask_init: float = 1.0) -> None:
        super().__init__()

        self.activation = nn.GELU()

        self.layer1 = nn.Linear(width, width)
        self.layer2 = nn.Linear(width, width)

        self.mask1 = Mask(width, mask_init)
        self.mask2 = Mask(width, mask_init)

        self.reset_parameters()

    def reset_parameters(self) -> None:
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
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_blocks: int = 3,
        mask_init: float = 1.0,
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
            betas=(0.9, 0.999),
        )

        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=config.scheduler_gamma,
        )

        self.iteration = 0
        self.history: list[tuple[int, float, float]] = []

    def _tensor(
        self,
        array: np.ndarray,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        return torch.tensor(
            array,
            dtype=self.config.dtype,
            device=self.device,
            requires_grad=requires_grad,
        )

    def solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, t], dim=1))

    def residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = self.solution(x, t)

        u_t = gradient(u, t)
        u_tt = gradient(u_t, t)

        u_x = gradient(u, x)
        u_xx = gradient(u_x, x)

        return u_tt - (self.config.c**2) * u_xx

    def sample_training_points(self) -> tuple[torch.Tensor, ...]:

        cfg = self.config

        x_initial = cfg.upper_bound_x * lhs(1, cfg.num_initial_points)

        x_0 = self._tensor(x_initial, requires_grad=True)
        t_0 = self._tensor(np.zeros_like(x_initial), requires_grad=True)

        t_boundary = cfg.upper_bound_t * lhs(1, cfg.num_boundary_points)

        x_left = self._tensor(
            np.full_like(t_boundary, cfg.lower_bound_x),
            requires_grad=True,
        )
        t_left = self._tensor(t_boundary, requires_grad=True)

        x_right = self._tensor(
            np.full_like(t_boundary, cfg.upper_bound_x),
            requires_grad=True,
        )
        t_right = self._tensor(t_boundary, requires_grad=True)

        x_collocation_np = (
            cfg.lower_bound_x
            + (cfg.upper_bound_x - cfg.lower_bound_x)
            * lhs(1, cfg.num_collocation_points)
        )

        t_collocation_np = (
            cfg.lower_bound_t
            + (cfg.upper_bound_t - cfg.lower_bound_t)
            * lhs(1, cfg.num_collocation_points)
        )

        x_collocation = self._tensor(
            x_collocation_np,
            requires_grad=True,
        )

        t_collocation = self._tensor(
            t_collocation_np,
            requires_grad=True,
        )

        return (
            x_0,
            t_0,
            x_left,
            t_left,
            x_right,
            t_right,
            x_collocation,
            t_collocation,
        )

    def compute_loss(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        (
            x_0,
            t_0,
            x_left,
            t_left,
            x_right,
            t_right,
            x_collocation,
            t_collocation,
        ) = self.sample_training_points()

        residual_loss = torch.mean(
            self.residual(x_collocation, t_collocation) ** 2
        )

        u_initial = self.solution(x_0, t_0)

        initial_loss = torch.mean(
            (u_initial - torch.sin(torch.pi * x_0)) ** 2
            + gradient(u_initial, t_0) ** 2
        )

        boundary_loss = torch.mean(
            self.solution(x_left, t_left) ** 2
            + self.solution(x_right, t_right) ** 2
        )

        total_loss = residual_loss + initial_loss + boundary_loss

        return total_loss, residual_loss, initial_loss

    @torch.no_grad()
    def relative_l2_error(self) -> float:

        prediction = to_numpy(
            self.solution(self.x_eval, self.t_eval)
        )

        return float(
            np.linalg.norm(self.exact_eval - prediction, 2)
            / np.linalg.norm(self.exact_eval, 2)
        )

    def train_step(self) -> tuple[float, float, float]:

        self.optimizer.zero_grad(set_to_none=True)

        total_loss, residual_loss, initial_loss = self.compute_loss()

        total_loss.backward()

        self.optimizer.step()

        self.iteration += 1

        return (
            float(total_loss.item()),
            float(residual_loss.item()),
            float(initial_loss.item()),
        )

    def train(self) -> np.ndarray:

        trainable_params = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        print(f"Device: {self.device}")
        print(f"Trainable parameters: {trainable_params:,}")

        self.model.train()

        for epoch in range(1, self.config.epochs + 1):

            start_time = time.time()

            loss, residual_loss, initial_loss = self.train_step()

            iteration_time = time.time() - start_time

            if epoch % self.config.scheduler_step_size == 0:
                self.scheduler.step()

            if self.iteration % self.config.log_every == 0:

                rel_l2 = self.relative_l2_error()

                self.history.append(
                    (self.iteration, rel_l2, loss)
                )

                print(
                    f"Iter {self.iteration:6d} | "
                    f"loss {loss:.3e} | "
                    f"residual {residual_loss:.3e} | "
                    f"initial {initial_loss:.3e} | "
                    f"rel_L2 {rel_l2:.3e} | "
                    f"time/iter {iteration_time:.2e}s"
                )

                self.save_history()

        return self.save_history()

    def save_history(self) -> np.ndarray:

        history = np.asarray(self.history, dtype=np.float64)

        np.savetxt(
            self.output_dir / "losses.txt",
            history,
            fmt="%.10f %.10f %.10f",
            header="iter rel_l2 loss",
        )

        return history

    def save_checkpoint(self, path: str | Path) -> None:

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

    def predict(
        self,
        x: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:

        self.model.eval()

        with torch.no_grad():

            x_tensor = self._tensor(x)
            t_tensor = self._tensor(t)

            prediction = self.solution(x_tensor, t_tensor)

        return to_numpy(prediction)
