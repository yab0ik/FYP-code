"""Train and save the baseline Black-Scholes neural network."""

import argparse
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
) -> dict[str, Any]:
    nn_config = _nn_config(config)
    used = {
        "hidden_sizes": list(hidden_sizes) if hidden_sizes is not None else list(nn_config.get("hidden_sizes", [32, 32])),
        "activation": activation if activation is not None else nn_config.get("activation", "relu"),
        "learning_rate": learning_rate if learning_rate is not None else nn_config.get("learning_rate", 0.001),
        "batch_size": batch_size if batch_size is not None else nn_config.get("batch_size", 256),
        "epochs": epochs if epochs is not None else nn_config.get("epochs", 100),
        "seed": seed if seed is not None else nn_config.get("seed", 42),
    }
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
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(used["batch_size"]),
        shuffle=True,
    )
    val_tensor = torch.from_numpy(val_x)
    val_target = torch.from_numpy(val_y)

    output_dir = Path("results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")

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
            val_loss = float(criterion(model(val_tensor), val_target).item())
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

    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_weights(model, output_dir / "weights.npz")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(train_losses, label="Train MSE")
    axis.plot(val_losses, label="Validation MSE")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.set_title(f"Loss curves: {run_name}")
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=Path("data/black_scholes_dataset.npz"))
    parser.add_argument("--run-name", default="b2_baseline")
    args = parser.parse_args()
    result = train_model(args.config, args.dataset, args.run_name)
    print(f"Best validation MSE: {result['best_validation_loss']:.10f}")
    print(f"Saved run to {result['run_dir']}")


if __name__ == "__main__":
    main()
