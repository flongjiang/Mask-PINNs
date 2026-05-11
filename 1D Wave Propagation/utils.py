"""Core code for a Mask-PINN solving the 1D wave equation."""

import random
import time
from pathlib import Path

import numpy as np
import torch
from pyDOE import lhs
from torch import nn
from torch.optim import lr_scheduler


C = 1.0
X_LOWER = 0.0
X_UPPER = 1.0
T_LOWER = 0.0
T_UPPER = 5.0

HIDDEN_DIM = 64
NUM_BLOCKS = 3
MASK_INIT = 1.0

NUM_INITIAL_POINTS = 400
NUM_BOUNDARY_POINTS = 400
NUM_COLLOCATION_POINTS = 5000
GRID_SIZE = 200

LEARNING_RATE = 1.0e-3
SCHEDULER_GAMMA = 0.9
SCHEDULER_STEP_SIZE = 500
LOG_EVERY = 200

DTYPE = torch.float32


def get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device)

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return requested


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tonp(tensor):
    return tensor.detach().cpu().numpy()


def grad(u, x):
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]


def build_reference_solution():
    x = np.linspace(X_LOWER, X_UPPER, GRID_SIZE, dtype=np.float32)
    t = np.linspace(T_LOWER, T_UPPER, GRID_SIZE, dtype=np.float32)

    x_grid, t_grid = np.meshgrid(x, t)
    exact = np.sin(np.pi * x_grid) * np.cos(C * np.pi * t_grid)

    return (
        x_grid.reshape(-1, 1),
        t_grid.reshape(-1, 1),
        exact.reshape(-1, 1).astype(np.float32),
    )


class Mask(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.alpha = nn.Parameter(MASK_INIT * torch.ones(width, dtype=DTYPE))

    def forward(self, x):
        return 1.0 - torch.exp(-((self.alpha * x) ** 2))


class MaskBlock(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.activation = nn.GELU()
        self.layer1 = nn.Linear(width, width)
        self.layer2 = nn.Linear(width, width)
        self.mask1 = Mask(width)
        self.mask2 = Mask(width)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, inputs):
        hidden = self.layer1(inputs)
        hidden = self.activation(hidden) * self.mask1(hidden)

        hidden = self.layer2(hidden)
        hidden = self.activation(hidden) * self.mask2(hidden)

        return hidden + inputs


class MaskPINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.GELU()
        self.input_layer = nn.Linear(2, HIDDEN_DIM)
        self.input_mask = Mask(HIDDEN_DIM)
        self.blocks = nn.Sequential(*[MaskBlock(HIDDEN_DIM) for _ in range(NUM_BLOCKS)])
        self.output_layer = nn.Linear(HIDDEN_DIM, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, coordinates):
        features = self.input_layer(coordinates)
        features = self.activation(features) * self.input_mask(features)
        features = self.blocks(features)
        return self.output_layer(features)


class PhysicsInformedNN:
    def __init__(self, x_eval, t_eval, exact_eval, output_dir, device, epochs):
        self.device = device
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exact_eval = exact_eval.astype(np.float32)
        self.x_eval = self.tensor(x_eval)
        self.t_eval = self.tensor(t_eval)

        self.model = MaskPINN().to(device=self.device, dtype=DTYPE)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
        )
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=SCHEDULER_GAMMA,
        )

        self.iteration = 0
        self.history = []

    def tensor(self, array, requires_grad=False):
        return torch.tensor(
            array,
            dtype=DTYPE,
            device=self.device,
            requires_grad=requires_grad,
        )

    def net_u(self, x, t):
        return self.model(torch.cat((x, t), dim=1))

    def net_r(self, x, t):
        u = self.net_u(x, t)

        u_t = grad(u, t)
        u_tt = grad(u_t, t)

        u_x = grad(u, x)
        u_xx = grad(u_x, x)

        return u_tt - (C**2) * u_xx

    def sample_training_points(self):
        x0 = X_LOWER + (X_UPPER - X_LOWER) * lhs(1, NUM_INITIAL_POINTS).astype(np.float32)
        tb = T_LOWER + (T_UPPER - T_LOWER) * lhs(1, NUM_BOUNDARY_POINTS).astype(np.float32)
        xf = X_LOWER + (X_UPPER - X_LOWER) * lhs(1, NUM_COLLOCATION_POINTS).astype(np.float32)
        tf = T_LOWER + (T_UPPER - T_LOWER) * lhs(1, NUM_COLLOCATION_POINTS).astype(np.float32)

        return (
            self.tensor(x0, requires_grad=True),
            self.tensor(np.full_like(x0, T_LOWER), requires_grad=True),
            self.tensor(np.full_like(tb, X_LOWER), requires_grad=True),
            self.tensor(tb, requires_grad=True),
            self.tensor(np.full_like(tb, X_UPPER), requires_grad=True),
            self.tensor(tb, requires_grad=True),
            self.tensor(xf, requires_grad=True),
            self.tensor(tf, requires_grad=True),
        )

    def loss_func(self):
        x0, t0, xl, tl, xr, tr, xf, tf = self.sample_training_points()

        loss_r = torch.mean(self.net_r(xf, tf) ** 2)

        u0 = self.net_u(x0, t0)
        loss_0 = torch.mean(
            (u0 - torch.sin(torch.pi * x0)) ** 2
            + grad(u0, t0) ** 2
        )

        loss_b = torch.mean(
            self.net_u(xl, tl) ** 2
            + self.net_u(xr, tr) ** 2
        )

        loss = loss_r + loss_0 + loss_b

        return loss, loss_r, loss_0, loss_b

    def relative_l2_error(self):
        with torch.no_grad():
            pred = tonp(self.net_u(self.x_eval, self.t_eval))

        return np.linalg.norm(self.exact_eval - pred, 2) / np.linalg.norm(self.exact_eval, 2)

    def train_step(self):
        self.optimizer.zero_grad(set_to_none=True)

        loss, loss_r, loss_0, loss_b = self.loss_func()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1

        return loss.item(), loss_r.item(), loss_0.item(), loss_b.item()

    def train(self):
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Device: {self.device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print(f"Trainable parameters: {params:,}")

        self.model.train()

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            loss, loss_r, loss_0, loss_b = self.train_step()

            if epoch % SCHEDULER_STEP_SIZE == 0:
                self.scheduler.step()

            if self.iteration % LOG_EVERY == 0:
                rel_l2 = self.relative_l2_error()
                self.history.append((self.iteration, rel_l2, loss))

                print(
                    f"Iter {self.iteration:6d} | "
                    f"loss {loss:.3e} | residual {loss_r:.3e} | "
                    f"initial {loss_0:.3e} | boundary {loss_b:.3e} | "
                    f"rel_L2 {rel_l2:.3e} | time/iter {time.time() - start:.2e}s"
                )

                self.save_history()

        return self.save_history()

    def save_history(self):
        history = np.asarray(self.history, dtype=np.float32)

        if history.size == 0:
            history = np.empty((0, 3), dtype=np.float32)

        np.savetxt(
            self.output_dir / "losses.txt",
            history,
            fmt="%.10f %.10f %.10f",
            header="iter rel_l2 loss",
        )

        return history

    def save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "iteration": self.iteration,
            },
            path,
        )

    def predict(self, x, t):
        self.model.eval()

        with torch.no_grad():
            return tonp(self.net_u(self.tensor(x), self.tensor(t)))
