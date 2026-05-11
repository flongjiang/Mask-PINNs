"""Train a Mask-PINN for the 1D linear advection equation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from utils import AdvectionConfig, AdvectionPINN, get_device


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_exact_solution(nx: int, nt: int, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the exact solution grid for u(x,t)=sin(x-beta*t)."""

    x_grid = np.linspace(0.0, 2.0 * np.pi, nx)
    t_grid = np.linspace(0.0, 1.0, nt)
    xv, tv = np.meshgrid(x_grid, t_grid)
    exact_u = np.sin(xv - beta * tv)

    x_flat = xv.reshape(-1, 1)
    t_flat = tv.reshape(-1, 1)
    u_flat = exact_u.reshape(-1, 1)
    x_exact = np.hstack((x_flat, t_flat, u_flat))
    return xv, tv, exact_u, x_exact


def save_prediction_figure(
    output_path: Path,
    xv: np.ndarray,
    tv: np.ndarray,
    exact_u: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """Save exact, predicted, and absolute-error fields as one figure."""

    x_star = np.hstack((xv.reshape(-1, 1), tv.reshape(-1, 1)))
    pred_grid = griddata(x_star, prediction.reshape(-1), (xv, tv), method="cubic")
    error_grid = np.abs(pred_grid - exact_u)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), dpi=300, constrained_layout=True)
    panels = [
        (exact_u.T, "Exact $u(x,t)$"),
        (pred_grid.T, "Predicted $u(x,t)$"),
        (error_grid.T, "Absolute error"),
    ]

    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(
            data,
            interpolation="nearest",
            cmap="jet",
            extent=[0.0, 1.0, 0.0, 2.0 * np.pi],
            origin="lower",
            aspect="auto",
        )
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask-PINN for 1D linear advection.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[5, 0, 9, 42, 1979])
    parser.add_argument("--epochs", type=int, default=50_000)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--n-initial", type=int, default=200)
    parser.add_argument("--n-boundary", type=int, default=200)
    parser.add_argument("--n-collocation", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--double", action="store_true", help="Use torch.float64 instead of torch.float32.")
    parser.add_argument("--no-plot", action="store_true", help="Disable saving prediction figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    dtype = torch.float64 if args.double else torch.float32

    xv, tv, exact_u, x_exact = build_exact_solution(args.nx, args.nt, args.beta)
    x_star = np.hstack((xv.reshape(-1, 1), tv.reshape(-1, 1)))
    u_star = exact_u.reshape(-1, 1)

    for seed in args.seeds:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_seed(seed)

        run_dir = args.output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = AdvectionConfig(
            beta=args.beta,
            n_initial=args.n_initial,
            n_boundary=args.n_boundary,
            n_collocation=args.n_collocation,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            learning_rate=args.lr,
            epochs=args.epochs,
            print_every=args.print_every,
            checkpoint_every=args.checkpoint_every,
        )

        trainer = AdvectionPINN(
            x_exact=x_exact,
            config=config,
            output_dir=run_dir,
            device=device,
            dtype=dtype,
        )
        history = trainer.train()
        np.savetxt(run_dir / "history.txt", history, fmt="%.10f %.10f %.10f", header="iter rel_l2 loss", comments="")
        torch.save(trainer.model.state_dict(), run_dir / "model.pt")

        prediction = trainer.predict(x_star[:, 0:1], x_star[:, 1:2])
        error_u = np.linalg.norm(u_star - prediction, 2) / np.linalg.norm(u_star, 2)
        print(f"Seed {seed} final relative L2 error: {error_u:.6e}")

        np.save(run_dir / "prediction.npy", prediction)
        if not args.no_plot:
            save_prediction_figure(run_dir / "prediction.png", xv, tv, exact_u, prediction)


if __name__ == "__main__":
    main()
