"""Core code for a Mask-PINN solving the 2D Helmholtz equation."""

import random
import time
from pathlib import Path

import numpy as np
import torch
from pyDOE import lhs
from torch import nn
from torch.optim import lr_scheduler


A1 = 6.0
A2 = 6.0
K = 1.0

X_LOWER = -1.0
X_UPPER = 1.0
Y_LOWER = -1.0
Y_UPPER = 1.0

HIDDEN_DIM = 128
NUM_BLOCKS = 5
MASK_INIT = 20.0

NUM_BOUNDARY_POINTS = 100
NUM_COLLOCATION_POINTS = 10000
GRID_SIZE = 201

LEARNING_RATE = 1.0e-4
SCHEDULER_GAMMA = 0.9
SCHEDULER_STEP_SIZE = 1000
LOG_EVERY = 100

BOUNDARY_WEIGHT = 100.0
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
    y = np.linspace(Y_LOWER, Y_UPPER, GRID_SIZE, dtype=np.float32)

    x_grid, y_grid = np.meshgrid(x, y)
    exact = np.sin(A1 * np.pi * x_grid) * np.sin(A2 * np.pi * y_grid)

    x_eval, y_eval = np.meshgrid(
        x.reshape(-1, 1),
        y.reshape(-1, 1),
        indexing="ij",
    )

    exact_eval = np.reshape(exact, (GRID_SIZE, GRID_SIZE)).T

    return (
        x_eval.flatten()[:, None].astype(np.float32),
        y_eval.flatten()[:, None].astype(np.float32),
        exact_eval.flatten()[:, None].astype(np.float32),
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
    def __init__(self, x_eval, y_eval, exact_eval, output_dir, device, epochs):
        self.device = device
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exact_eval = exact_eval.astype(np.float32)
        self.x_eval = self.tensor(x_eval)
        self.y_eval = self.tensor(y_eval)

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

    def net_u(self, x, y):
        return self.model(torch.cat((x, y), dim=1))

    def net_r(self, x, y):
        forcing = (
            -((A1 * np.pi) ** 2)
            * torch.sin(A1 * np.pi * x)
            * torch.sin(A2 * np.pi * y)
            - ((A2 * np.pi) ** 2)
            * torch.sin(A1 * np.pi * x)
            * torch.sin(A2 * np.pi * y)
            + K
            * torch.sin(A1 * np.pi * x)
            * torch.sin(A2 * np.pi * y)
        )

        u = self.net_u(x, y)

        u_x = grad(u, x)
        u_y = grad(u, y)

        u_xx = grad(u_x, x)
        u_yy = grad(u_y, y)

        return u_xx + u_yy + K**2 * u - forcing

    def sample_training_points(self):
        xb = X_LOWER + (X_UPPER - X_LOWER) * lhs(1, NUM_BOUNDARY_POINTS).astype(np.float32)
        yb = Y_LOWER + (Y_UPPER - Y_LOWER) * lhs(1, NUM_BOUNDARY_POINTS).astype(np.float32)

        x_left = self.tensor(np.full_like(yb, X_LOWER), requires_grad=True)
        y_left = self.tensor(yb, requires_grad=True)

        x_right = self.tensor(np.full_like(yb, X_UPPER), requires_grad=True)
        y_right = self.tensor(yb, requires_grad=True)

        x_bottom = self.tensor(xb, requires_grad=True)
        y_bottom = self.tensor(np.full_like(xb, Y_LOWER), requires_grad=True)

        x_top = self.tensor(xb, requires_grad=True)
        y_top = self.tensor(np.full_like(xb, Y_UPPER), requires_grad=True)

        collocation = np.array([X_LOWER, Y_LOWER], dtype=np.float32) + (
            np.array([X_UPPER - X_LOWER, Y_UPPER - Y_LOWER], dtype=np.float32)
            * lhs(2, NUM_COLLOCATION_POINTS).astype(np.float32)
        )

        x_f = self.tensor(collocation[:, 0:1], requires_grad=True)
        y_f = self.tensor(collocation[:, 1:2], requires_grad=True)

        return (
            x_left,
            y_left,
            x_right,
            y_right,
            x_bottom,
            y_bottom,
            x_top,
            y_top,
            x_f,
            y_f,
        )

    def loss_func(self):
        (
            x_left,
            y_left,
            x_right,
            y_right,
            x_bottom,
            y_bottom,
            x_top,
            y_top,
            x_f,
            y_f,
        ) = self.sample_training_points()

        loss_r = torch.mean(self.net_r(x_f, y_f) ** 2)

        loss_b = (
            torch.mean(self.net_u(x_left, y_left) ** 2)
            + torch.mean(self.net_u(x_right, y_right) ** 2)
            + torch.mean(self.net_u(x_bottom, y_bottom) ** 2)
            + torch.mean(self.net_u(x_top, y_top) ** 2)
        )

        loss = loss_r + BOUNDARY_WEIGHT * loss_b

        return loss, loss_r, loss_b

    def relative_l2_error(self):
        with torch.no_grad():
            pred = tonp(self.net_u(self.x_eval, self.y_eval))

        return np.linalg.norm(self.exact_eval - pred, 2) / np.linalg.norm(self.exact_eval, 2)

    def train_step(self):
        self.optimizer.zero_grad(set_to_none=True)

        loss, loss_r, loss_b = self.loss_func()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1

        return loss.item(), loss_r.item(), loss_b.item()

    def train(self):
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Device: {self.device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print(f"Trainable parameters: {params:,}")

        self.model.train()

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            loss, loss_r, loss_b = self.train_step()

            if epoch % SCHEDULER_STEP_SIZE == 0:
                self.scheduler.step()

            if self.iteration % LOG_EVERY == 0:
                rel_l2 = self.relative_l2_error()
                self.history.append((self.iteration, rel_l2, loss))

                print(
                    f"Iter {self.iteration:6d} | "
                    f"loss {loss:.3e} | residual {loss_r:.3e} | "
                    f"boundary {loss_b:.3e} | "
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

    def predict(self, x, y):
        self.model.eval()

        with torch.no_grad():
            return tonp(self.net_u(self.tensor(x), self.tensor(y)))
