"""Train a Mask-PINN on a 2D Helmholtz benchmark.

Example:
    python train.py --epochs 50000 --seeds 42 0 9 5 1979 --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from utils import PhysicsInformedNN, TrainConfig, build_reference_solution, get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask-PINN for a 2D Helmholtz problem.")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 9, 5, 1979])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--num-boundary-points", type=int, default=100)
    parser.add_argument("--num-collocation-points", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = torch.float64

    config = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_boundary_points=args.num_boundary_points,
        num_collocation_points=args.num_collocation_points,
        grid_size=args.grid_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        dtype=dtype,
    )
    x_eval, y_eval, exact_eval = build_reference_solution(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seed_everything(seed)

        seed_dir = args.output_dir / f"seed_{seed}"
        trainer = PhysicsInformedNN(
            config=config,
            x_eval=x_eval,
            y_eval=y_eval,
            exact_eval=exact_eval,
            output_dir=seed_dir,
            device=device,
        )
        history = trainer.train()
        trainer.save_checkpoint(seed_dir / "model.pt")
        np.savetxt(seed_dir / f"losses_{seed}.txt", history, fmt="%.10f %.10f %.10f", header="iter rel_l2 loss")


if __name__ == "__main__":
    main()
