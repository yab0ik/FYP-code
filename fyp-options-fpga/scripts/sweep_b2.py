"""Sweep B2 baseline neural-network configurations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from optfpga.nn.evaluate import evaluate
from optfpga.nn.model import MLP
from optfpga.nn.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/black_scholes_dataset.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    hidden_grid = [[16], [16, 16], [32, 32], [64, 64]]
    activation_grid = ["relu", "tanh"]
    sweep_dir = Path("results") / "b2_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for hidden_sizes in hidden_grid:
        for activation in activation_grid:
            run_name = f"b2_sweep/{'-'.join(map(str, hidden_sizes))}_{activation}"
            result = train_model(
                args.config,
                args.dataset,
                run_name,
                hidden_sizes=hidden_sizes,
                activation=activation,
                epochs=args.epochs,
            )

            run_dir = Path("results") / run_name
            checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
            model = MLP(checkpoint["input_size"], checkpoint["hidden_sizes"], checkpoint["activation"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with np.load(args.dataset) as data:
                mean = data["X_reduced_mean"].astype(np.float32)
                std = data["X_reduced_std"].astype(np.float32)
                val_x = ((data["X_reduced_val"] - mean) / std).astype(np.float32)
                val_y = data["call_over_K_val"].reshape(-1).astype(np.float64)
            with torch.no_grad():
                val_pred = model(torch.from_numpy(val_x)).numpy().reshape(-1).astype(np.float64)
            val_errors = val_pred - val_y
            rows.append({
                "hidden_sizes": json.dumps(hidden_sizes),
                "activation": activation,
                "n_parameters": result["n_parameters"],
                "validation_mse": result["best_validation_loss"],
                "validation_mae": float(np.mean(np.abs(val_errors))),
                "validation_max_abs_error": float(np.max(np.abs(val_errors))),
            })

    csv_path = sweep_dir / "validation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "hidden_sizes",
                "activation",
                "n_parameters",
                "validation_mse",
                "validation_mae",
                "validation_max_abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_mse = min(row["validation_mse"] for row in rows)
    mse_limit = best_mse * (1.0 + args.tolerance)
    eligible = [row for row in rows if row["validation_mse"] <= mse_limit]
    selected = min(
        eligible,
        key=lambda row: (row["n_parameters"], row["validation_mse"], row["activation"]),
    )
    selected_hidden = json.loads(selected["hidden_sizes"])
    selected_activation = selected["activation"]
    print(
        f"Selected: hidden_sizes={selected_hidden}, activation={selected_activation}. "
        f"Why: {selected['n_parameters']} parameters, validation MSE={selected['validation_mse']:.8f}; "
        f"it is the smallest network within {args.tolerance:.1%} "
        f"of the best validation MSE={best_mse:.8f} (limit={mse_limit:.8f})."
    )

    base_seed = 42
    seed_metrics = []
    for offset in range(3):
        seed = base_seed + offset
        run_name = f"b2_selected_seed_{seed}"
        train_model(
            args.config,
            args.dataset,
            run_name,
            hidden_sizes=selected_hidden,
            activation=selected_activation,
            seed=seed,
            epochs=args.epochs,
        )
        metrics = evaluate(
            Path("results") / run_name,
            args.dataset,
            Path("results") / run_name / "metrics.json",
        )
        seed_metrics.append(metrics)

    metric_names = [
        "mae_call_over_K",
        "rmse_call_over_K",
        "max_abs_error_call_over_K",
        "mae_price",
        "rmse_price",
        "max_abs_error_price",
        "fraction_price_error_above_0.01",
        "fraction_price_error_above_0.1",
    ]
    spread = {
        name: {
            "mean": float(np.mean([m[name] for m in seed_metrics])),
            "min": float(np.min([m[name] for m in seed_metrics])),
            "max": float(np.max([m[name] for m in seed_metrics])),
            "std": float(np.std([m[name] for m in seed_metrics], ddof=0)),
        }
        for name in metric_names
    }
    (sweep_dir / "selected_seed_spread.json").write_text(
        json.dumps({
            "hidden_sizes": selected_hidden,
            "activation": selected_activation,
            "seeds": [base_seed, base_seed + 1, base_seed + 2],
            "metrics": spread,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "seed_spread": spread}, indent=2))


if __name__ == "__main__":
    main()
