"""Train Mask-PINN for 1D advection."""

import argparse
from pathlib import Path

import numpy as np
import torch

from utils import PhysicsInformedNN, build_reference_solution, get_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5, 0, 9, 42, 1979])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    x_eval, t_eval, exact_eval = build_reference_solution()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_everything(seed)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        seed_dir = args.output_dir / f"seed_{seed}"

        model = PhysicsInformedNN(
            x_eval=x_eval,
            t_eval=t_eval,
            exact_eval=exact_eval,
            output_dir=seed_dir,
            device=device,
            epochs=args.epochs,
        )

        history = model.train()

        model.save_checkpoint(seed_dir / "model.pt")

        np.savetxt(
            seed_dir / f"losses_{seed}.txt",
            history,
            fmt="%.10f %.10f %.10f",
            header="iter rel_l2 loss",
        )


if __name__ == "__main__":
    main()
