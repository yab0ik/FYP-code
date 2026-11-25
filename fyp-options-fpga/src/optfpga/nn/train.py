"""Train and save the baseline Black-Scholes neural network."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optfpga.config import load_config
from optfpga.nn.model import MLP

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _nn_config(config: dict[str, Any]) -> dict[str, Any]:
    nn_config = config.get("nn")
    if not isinstance(nn_config, dict):
        nn_config = {}
    return nn_config


def training_config(
    config: dict[str, Any],
    *,
    hidden_sizes: Iterable[int] | None = None,
    activation: str | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    lr_schedule: str | None = None,
) -> dict[str, Any]:
    nn_config = _nn_config(config)
    used = {
        "hidden_sizes": list(hidden_sizes) if hidden_sizes is not None else list(nn_config.get("hidden_sizes", [32, 32])),
        "activation": activation if activation is not None else nn_config.get("activation", "relu"),
        "learning_rate": learning_rate if learning_rate is not None else nn_config.get("learning_rate", 0.001),
        "batch_size": batch_size if batch_size is not None else nn_config.get("batch_size", 256),
        "epochs": epochs if epochs is not None else nn_config.get("epochs", 100),
        "seed": seed if seed is not None else nn_config.get("seed", 42),
        "lr_schedule": lr_schedule if lr_schedule is not None else nn_config.get("lr_schedule"),
    }
    if used["lr_schedule"] not in (None, "cosine"):
        raise ValueError("lr_schedule must be None or 'cosine'")
    return used


def save_weights(model: MLP, output_path: str | Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            linear_index = sum(isinstance(item, nn.Linear) for item in model.network[: index + 1]) - 1
            arrays[f"weight_{linear_index}"] = layer.weight.detach().cpu().numpy()
            arrays[f"bias_{linear_index}"] = layer.bias.detach().cpu().numpy()
    np.savez(output_path, **arrays)


def train_model(
    config_path: str | Path | None = None,
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
    run_name: str = "b2_baseline",
    *,
    hidden_sizes: Iterable[int] | None = None,
    activation: str | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    lr_schedule: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    used = training_config(
        config,
        hidden_sizes=hidden_sizes,
        activation=activation,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        lr_schedule=lr_schedule,
    )
    set_seed(int(used["seed"]))

    with np.load(dataset_path) as data:
        mean = data["X_reduced_mean"].astype(np.float32)
        std = data["X_reduced_std"].astype(np.float32)
        train_x = ((data["X_reduced_train"] - mean) / std).astype(np.float32)
        train_y = data["call_over_K_train"].reshape(-1, 1).astype(np.float32)
        val_x = ((data["X_reduced_val"] - mean) / std).astype(np.float32)
        val_y = data["call_over_K_val"].reshape(-1, 1).astype(np.float32)

    model = MLP(4, used["hidden_sizes"], used["activation"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(used["learning_rate"]))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(used["epochs"]),
        )
        if used["lr_schedule"] == "cosine"
        else None
    )
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(used["batch_size"]),
        shuffle=True,
    )
    val_tensor = torch.from_numpy(val_x)
    val_target = torch.from_numpy(val_y)

    run_path = Path(run_name)
    output_dir = run_path if run_path.is_absolute() else Path("results") / run_path
    output_dir.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    final_val_predictions: np.ndarray | None = None

    for epoch in range(int(used["epochs"])):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_tensor)
            val_loss = float(criterion(val_predictions, val_target).item())
            if epoch == int(used["epochs"]) - 1:
                final_val_predictions = val_predictions.numpy().reshape(-1).astype(np.float64)
        train_losses.append(epoch_loss / seen)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": 4,
                    "hidden_sizes": used["hidden_sizes"],
                    "activation": used["activation"],
                },
                output_dir / "model.pt",
            )
        if scheduler is not None:
            scheduler.step()

    if final_val_predictions is None:
        raise RuntimeError("Training completed without final validation predictions")
    final_val_errors = final_val_predictions - val_y.reshape(-1).astype(np.float64)
    final_validation_mse = val_losses[-1]
    final_validation_mae = float(np.mean(np.abs(final_val_errors)))
    final_validation_max_abs_error = float(np.max(np.abs(final_val_errors)))

    used_config = dict(config)
    used_config["nn"] = used
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(used_config, sort_keys=False),
        encoding="utf-8",
    )
    np.savez(
        output_dir / "normalization.npz",
        X_reduced_mean=mean,
        X_reduced_std=std,
    )
    run_label = Path(run_name).name
    history = {
        "epochs": list(range(1, len(train_losses) + 1)),
        "train_losses": train_losses,
        "validation_losses": val_losses,
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )

    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_weights(model, output_dir / "weights.npz")

    figure, axis = plt.subplots(figsize=(8, 5))
    epoch_numbers = range(1, len(train_losses) + 1)
    axis.plot(epoch_numbers, train_losses, label="Train MSE")
    axis.plot(epoch_numbers, val_losses, label="Validation MSE")
    best_epoch = int(np.argmin(val_losses)) + 1
    axis.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best validation epoch ({best_epoch})",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.set_title(f"Loss curves: {run_label} (best validation MSE={best_val:.10g})")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss.png", dpi=160)
    plt.close(figure)

    return {
        "run_dir": output_dir,
        "best_validation_loss": best_val,
        "train_losses": train_losses,
        "validation_losses": val_losses,
        "config": used,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_epoch": int(np.argmin(val_losses)) + 1,
        "final_validation_mse": final_validation_mse,
        "final_validation_mae": final_validation_mae,
        "final_validation_max_abs_error": final_validation_max_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=Path("data/black_scholes_dataset.npz"))
    parser.add_argument("--run-name", default="b2_baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr-schedule", choices=["cosine"], default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--activation", choices=["relu", "tanh"], default=None)
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    config_path = args.config
    if config_path is not None and not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_name = Path(args.run_name)
    if not run_name.is_absolute():
        run_name = PROJECT_ROOT / "results" / run_name

    result = train_model(
        config_path,
        dataset_path,
        str(run_name),
        epochs=args.epochs,
        seed=args.seed,
        lr_schedule=args.lr_schedule,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
    )
    print(f"Best validation MSE: {result['best_validation_loss']:.10f}")
    if len(result["validation_losses"]) > 100:
        best_first_100 = min(result["validation_losses"][:100])
        best_after_100 = min(result["validation_losses"][100:])
        print(
            f"Validation MSE after epoch 100: "
            f"{'improved' if best_after_100 < best_first_100 else 'did not improve'} "
            f"(best through epoch 100={best_first_100:.10f}, "
            f"best after epoch 100={best_after_100:.10f})."
        )
    best_epoch = result["best_validation_epoch"]
    epoch_index = best_epoch - 1
    print(f"Best validation epoch: {best_epoch}")
    print(f"Train MSE at best validation epoch: {result['train_losses'][epoch_index]:.10f}")
    print(f"Validation MSE at best validation epoch: {result['validation_losses'][epoch_index]:.10f}")
    print(f"Saved run to {result['run_dir']}")


if __name__ == "__main__":
    main()
