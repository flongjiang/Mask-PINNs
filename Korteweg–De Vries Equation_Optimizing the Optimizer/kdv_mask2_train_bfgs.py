# kdv_mask2_train_bfgs.py
# ------------------------------------------------------------
# KdV PINN with Mask2 template (ADAM warmup + SciPy-BFGS)
#
# - PDE (same as your utils.py):  u_t + u*u_x + (0.022)^2 * u_xxx = 0
# - IC  (same as your utils.py):  u(x,0) = cos(pi x)
# - Sampling (same style as your utils.py): LHS in (x,t)
# - Optimizer protocol (same as your mask2.py): ADAM -> (SciPy) BFGS
#
# Output:
#   results_kdv_mask2/
#     train_log.txt
#     losses_adam.npy
#     losses_bfgs.npy
#     relL2_adam.npy
#     relL2_bfgs.npy
#     model_final.pth
#     bfgs_resume_ckpt.pth
#     bfgs_ckpt_itXXXXXXX.pth      (SIMPLE ckpt every 1000 BFGS iters, NO overwrite)
#     pred_vs_exact.png
# ------------------------------------------------------------

import os
import time
import math
from time import perf_counter

import numpy as np
import scipy.io
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pyDOE import lhs
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------
# Global settings
# -----------------------------
torch.set_default_dtype(torch.float64)
torch.manual_seed(2)
np.random.seed(2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

RESULTS_DIR = "results_kdv_mask2"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, "train_log.txt")
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("# phase,iter,loss,relL2,lr,elapsed_sec\n")

def log_txt(phase: str, it: int, loss_val: float, relL2: float, lr: float, elapsed_sec: float):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{phase},{it},{loss_val:.12e},{relL2:.12e},{lr:.12e},{elapsed_sec:.6f}\n")
        f.flush()

# -----------------------------
# Problem: KdV
# -----------------------------
lb = np.array([-1.0, 0.0])  # x,t
ub = np.array([ 1.0, 1.0])

LAM_IC = 100.0
kdv_alpha = (0.022 ** 2)

N0  = 500
N_f = 5000

Nepochs_ADAM = 20000
Nchange      = 2000
Nprint       = 1000

Nbfgs        = 6000
BFGS_chunk   = 2000
BFGS_print   = 100

# >>> simple ckpt every N BFGS iters (NO overwrite)
BFGS_CKPT_EVERY = 1000

hidden = 128
n_blocks = 2
ff_in_dim = 3

# -----------------------------
# RESUME settings (BFGS)
# -----------------------------
RESUME_BFGS = False
RESUME_CKPT = os.path.join("bfgs_ckpt_it0004000.pth")  # <-- change if needed

# -----------------------------
# Utilities
# -----------------------------
def tonp(t):
    return t.detach().cpu().numpy()

def grad(u, x):
    g = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]
    if g is None:
        return torch.zeros_like(x)
    return g

def input_encoding(x, t):
    L = 2.0
    w = 2.0 * math.pi / L
    w = torch.tensor(w, dtype=torch.get_default_dtype(), device=x.device)
    return torch.cat((t, torch.cos(w * x), torch.sin(w * x)), 1)

def sample_X0(N0: int):
    x0 = -2.0 * lhs(1, N0) + 1.0
    t0 = np.zeros_like(x0)
    X0 = np.hstack([x0, t0])
    return torch.as_tensor(X0, dtype=torch.get_default_dtype(), device=DEVICE)

def sample_Xf(Nf: int):
    Xf = lb + (ub - lb) * lhs(2, Nf)
    return torch.as_tensor(Xf, dtype=torch.get_default_dtype(), device=DEVICE)

# -----------------------------
# Mask2 blocks
# -----------------------------
class MaskGate(nn.Module):
    """m(z)=1-exp(-(a*z)^2), with 'a' per hidden dim."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(hidden_dim, dtype=torch.get_default_dtype()))

    def forward(self, z):
        h = 1.0 * self.a * z
        return 1.0 - torch.exp(-(h ** 2))

class MaskBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.act = nn.Tanh()
        self.m1 = MaskGate(width)
        self.m2 = MaskGate(width)
        self.l1 = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)

    def forward(self, H):
        x1_0 = self.l1(H)
        x1 = self.act(x1_0)
        x2_0 = self.l2(x1 * self.m1(x1_0))
        x2 = self.act(x2_0)
        return x2 * self.m2(x2_0) + H

class Mask2KdVNet(nn.Module):
    """
    (x,t) -> encoding -> Fourier features -> mask blocks -> out
    """
    def __init__(self, hidden=128, n_blocks=4):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks

        self.W = nn.Parameter(
            2.0 * torch.randn(ff_in_dim, hidden // 2, dtype=torch.get_default_dtype()),
            requires_grad=True
        )

        self.blocks = nn.Sequential(*[MaskBlock(hidden) for _ in range(n_blocks)])
        self.out_linear = nn.Linear(hidden, 1)
        self._init_last_layer(scale=1.0)

    def _init_last_layer(self, scale: float):
        fan_in, fan_out = self.out_linear.in_features, self.out_linear.out_features
        n = 0.5 * (fan_in + fan_out)
        limit = math.sqrt(3.0 * scale / n)
        with torch.no_grad():
            self.out_linear.weight.uniform_(-limit, limit)
            if self.out_linear.bias is not None:
                self.out_linear.bias.zero_()

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        H = input_encoding(x, t)      # [N,3]
        Z = H @ self.W                # [N, hidden//2]
        Phi = torch.cat([torch.sin(Z), torch.cos(Z)], dim=1)  # [N, hidden]

        H0 = Phi
        Hh = torch.tanh(H0)
        Hh = self.blocks(Hh)
        out = self.out_linear(Hh)
        return out

# -----------------------------
# PDE + loss
# -----------------------------
def u_ic(x):
    return torch.cos(math.pi * x)

def net_u(model, X):
    return model(X)

def net_r(model, X):
    X = X.detach()
    x = X[:, 0:1].clone().detach().requires_grad_(True)
    t = X[:, 1:2].clone().detach().requires_grad_(True)
    Xt = torch.cat([x, t], dim=1)

    u = net_u(model, Xt)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    u_xxx = grad(u_xx, x)

    f = u_t + u * u_x + kdv_alpha * u_xxx
    return f

def loss_total(model, Xf, X0):
    r = net_r(model, Xf)
    loss_r = torch.mean(r**2)

    u0_pred = net_u(model, X0)
    u0_true = u_ic(X0[:, 0:1])
    loss_0 = torch.mean((u0_pred - u0_true)**2)

    return loss_r + LAM_IC * loss_0, loss_r, loss_0

# -----------------------------
# Exact data for rel-L2
# -----------------------------
def load_kdv_mat(path="kdv.mat"):
    data = scipy.io.loadmat(path)
    Exact = np.real(data["usol"])
    t_sol = data["t"].flatten()[:, None]
    x_sol = data["x"].flatten()[:, None]
    X, T = np.meshgrid(x_sol, t_sol)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    return Exact, x_sol, t_sol, X_star

def rel_l2(model, Exact, X_star):
    model.eval()
    with torch.no_grad():
        X = torch.as_tensor(X_star, dtype=torch.get_default_dtype(), device=DEVICE)
        u_pred = net_u(model, X).reshape(Exact.shape)
    model.train()
    u_np = u_pred.detach().cpu().numpy()
    return np.linalg.norm(Exact - u_np) / np.linalg.norm(Exact)

# -----------------------------
# SciPy BFGS bridge
# -----------------------------
power = 1.0

def loss_and_grad_torch(model, Xf, X0, power=1.0):
    L, _, _ = loss_total(model, Xf, X0)
    L_root = L if power == 1.0 else L ** (1.0 / power)
    params = list(model.parameters())
    grads_list = torch.autograd.grad(L_root, params, create_graph=False, retain_graph=False)
    gflat = torch.cat([g.reshape(-1) for g in grads_list])
    return L_root, gflat

def loss_and_grad_np(w_np, model, Xf, X0):
    w = torch.as_tensor(w_np, dtype=torch.get_default_dtype(), device=DEVICE)
    with torch.no_grad():
        vector_to_parameters(w, model.parameters())
    L_root, gflat = loss_and_grad_torch(model, Xf, X0, power=power)
    return float(L_root.detach().cpu().item()), gflat.detach().cpu().numpy()

# -----------------------------
# Training
# -----------------------------
def main():
    Exact, x_sol, t_sol, X_star = load_kdv_mat("kdv.mat")

    model = Mask2KdVNet(hidden=hidden, n_blocks=n_blocks).to(DEVICE)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    X0 = sample_X0(N0)
    Xf = sample_Xf(N_f)

    # ---------------- ADAM warmup ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    step_size = 2000

    adam_losses = []
    adam_rel = []
    adam_t0 = perf_counter()

    for ep in range(1, Nepochs_ADAM + 1):
        if (ep % Nchange) == 0:
            X0 = sample_X0(N0)
            Xf = sample_Xf(N_f)

        optimizer.zero_grad(set_to_none=True)
        L, Lr, L0 = loss_total(model, Xf, X0)
        L.backward()
        optimizer.step()

        if (ep % step_size) == 0:
            scheduler.step()

        adam_losses.append(float(L.detach().cpu().item()))

        if (ep % Nprint) == 0 or ep == 1:
            l2 = rel_l2(model, Exact, X_star)
            adam_rel.append((ep, float(l2)))
            lr_now = float(optimizer.param_groups[0]["lr"])
            elapsed = perf_counter() - adam_t0
            print(f"[ADAM] ep={ep:6d}  loss={adam_losses[-1]:.3e}  relL2={l2:.3e}  lr={lr_now:.2e}")
            log_txt("adam", ep, float(adam_losses[-1]), float(l2), lr_now, float(elapsed))

    np.save(os.path.join(RESULTS_DIR, "losses_adam.npy"), np.asarray(adam_losses))
    np.save(os.path.join(RESULTS_DIR, "relL2_adam.npy"), np.asarray(adam_rel, dtype=float))

    # ---------------- BFGS (SciPy) ----------------
    initial_weights = parameters_to_vector([p.detach() for p in model.parameters()]).cpu().numpy()

    cont = 0
    bfgs_losses = []
    bfgs_rel = []

    H0 = np.eye(initial_weights.size, dtype=np.float64)
    initial_time_bfgs = perf_counter()

    # keep latest xk (for ckpt storing true current weights vector)
    last_xk = None

    class _Res:
        __slots__ = ("fun",)
        def __init__(self, fun): self.fun = fun

    # ---------- RESUME from your "simple ckpt" (cont/x/H0/model_state_dict) ----------
    if RESUME_BFGS:
        ckpt = torch.load(RESUME_CKPT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        cont = int(ckpt["cont"])
        initial_weights = np.asarray(ckpt["x"], dtype=np.float64)
        H0 = np.asarray(ckpt["H0"], dtype=np.float64)
        last_xk = np.array(initial_weights, copy=True)
        print(f"[RESUME] {RESUME_CKPT}")
        print(f"[RESUME] cont={cont}, n={initial_weights.size}, H0 shape={H0.shape}")

    # >>> SIMPLE ckpt saver (NO overwrite)  (SAME format as your original)
    def save_simple_bfgs_ckpt():
        nonlocal last_xk, H0, cont, initial_weights
        ckpt_name = f"bfgs_ckpt_it{cont:07d}.pth"
        ckpt_path = os.path.join(RESULTS_DIR, ckpt_name)
        x_to_save = np.asarray(last_xk if last_xk is not None else initial_weights, dtype=np.float64)
        torch.save({
            "cont": int(cont),
            "x": x_to_save,                 # current weights vector
            "H0": np.asarray(H0, dtype=np.float64),
            "model_state_dict": model.state_dict(),
        }, ckpt_path, pickle_protocol=4)
        print(f"[SAVE] simple BFGS ckpt -> {ckpt_path}")

    def callback(*, intermediate_result):
        nonlocal cont, H0
        cont += 1

        # save every 1000 BFGS iters (independent of print frequency)
        if (cont % BFGS_CKPT_EVERY) == 0:
            save_simple_bfgs_ckpt()

        # log every BFGS_print
        if (cont % BFGS_print) != 0:
            return

        loss_value = float((intermediate_result.fun) ** power)
        l2 = rel_l2(model, Exact, X_star)
        bfgs_losses.append(loss_value)
        bfgs_rel.append((cont, float(l2)))
        elapsed = perf_counter() - initial_time_bfgs
        print(f"[BFGS] it={cont:7d}  loss={loss_value:.3e}  relL2={l2:.3e}")
        log_txt("bfgs", cont, float(loss_value), float(l2), float("nan"), float(elapsed))

    def scipy_callback(xk):
        nonlocal last_xk, initial_weights
        last_xk = np.array(xk, copy=True)
        initial_weights = np.array(xk, copy=True)  # keep consistent for next chunk / ckpt
        f, _ = loss_and_grad_np(xk, model, Xf, X0)
        callback(intermediate_result=_Res(f))

    method = "BFGS"
    method_bfgs = "SSBroyden2"
    initial_scale = False

    bfgs_t0 = perf_counter()
    while cont < Nbfgs:
        result = minimize(
            fun=loss_and_grad_np,
            x0=initial_weights,
            args=(model, Xf, X0),
            method=method,
            jac=True,
            options={
                "maxiter": BFGS_chunk,
                "gtol": 0,
                "hess_inv0": H0,
                "method_bfgs": method_bfgs,
                "initial_scale": initial_scale,
            },
            tol=0,
            callback=scipy_callback,
        )

        initial_weights = np.asarray(result.x, dtype=np.float64)
        last_xk = np.array(initial_weights, copy=True)

        H0 = result.hess_inv
        H0 = 0.5 * (H0 + H0.T)

        try:
            cholesky(H0)
        except LinAlgError:
            H0 = np.eye(len(initial_weights), dtype=np.float64)

        X0 = sample_X0(N0)
        Xf = sample_Xf(N_f)
        initial_scale = False

    bfgs_time_sec = perf_counter() - bfgs_t0
    print(f"[DONE] ADAM+BFGS finished. Logs -> {LOG_PATH}")
    np.save(os.path.join(RESULTS_DIR, "losses_bfgs.npy"), np.asarray(bfgs_losses, dtype=float))
    np.save(os.path.join(RESULTS_DIR, "relL2_bfgs.npy"), np.asarray(bfgs_rel, dtype=float))

    # ---------------- Save model + resume ckpt (full info) ----------------
    MODEL_PATH = os.path.join(RESULTS_DIR, "model_final.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": hidden,
        "n_blocks": n_blocks,
        "LAM_IC": LAM_IC,
        "kdv_alpha": kdv_alpha,
        "lb": lb,
        "ub": ub,
    }, MODEL_PATH, pickle_protocol=4)
    print("[SAVE] Model:", MODEL_PATH)

    CKPT_BFGS_PATH = os.path.join(RESULTS_DIR, "bfgs_resume_ckpt.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "initial_weights": np.asarray(initial_weights, dtype=np.float64),
        "H0": np.asarray(H0, dtype=np.float64),
        "cont": int(cont),
        "X0": X0.detach().cpu(),
        "Xf": Xf.detach().cpu(),
    }, CKPT_BFGS_PATH, pickle_protocol=4)
    print("[SAVE] BFGS resume ckpt:", CKPT_BFGS_PATH)

    # ---------------- Plot ----------------
    model.eval()
    with torch.no_grad():
        X = torch.as_tensor(X_star, dtype=torch.get_default_dtype(), device=DEVICE)
        u_pred = net_u(model, X).detach().cpu().numpy().reshape(Exact.shape)
    model.train()

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams['figure.max_open_warning'] = 4

    fig, ax = plt.subplots(dpi=300)
    h = ax.imshow(u_pred.T, interpolation='nearest', cmap='jet',
                  extent=[0.0, 1.0, -1.0, 1.0],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_title("Predicted $u(x,t)$ (Mask2 + ADAM+BFGS)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pred_only.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(20, 3), dpi=300)
    plt.subplot(1, 3, 1)
    h = plt.imshow(Exact.T, interpolation='nearest', cmap='jet',
                   extent=[0.0, 1.0, -1.0, 1.0],
                   origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 14})
    plt.ylabel(r'$x$', fontdict={'fontsize': 14})
    plt.title("Exact $u(x,t)$", fontdict={'fontsize': 14})

    plt.subplot(1, 3, 2)
    h = plt.imshow(u_pred.T, interpolation='nearest', cmap='jet',
                   extent=[0.0, 1.0, -1.0, 1.0],
                   origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 14})
    plt.ylabel(r'$x$', fontdict={'fontsize': 14})
    plt.title("Predicted $u(x,t)$", fontdict={'fontsize': 14})

    plt.subplot(1, 3, 3)
    h = plt.imshow(np.abs(u_pred - Exact).T, interpolation='nearest', cmap='jet',
                   extent=[0.0, 1.0, -1.0, 1.0],
                   origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 14})
    plt.ylabel(r'$x$', fontdict={'fontsize': 14})
    plt.title("Absolute error", fontdict={'fontsize': 14})

    plt.tight_layout()
    outfig = os.path.join(RESULTS_DIR, "pred_vs_exact.png")
    plt.savefig(outfig, dpi=300)
    plt.show()
    print("[FIG] saved:", outfig)

if __name__ == "__main__":
    main()
