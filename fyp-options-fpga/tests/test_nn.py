import numpy as np
import torch
from pathlib import Path

from optfpga.nn.evaluate import evaluate_test
from optfpga.nn.model import MLP
from optfpga.nn.train import set_seed, train_model


def test_model_output_shape_and_determinism() -> None:
    set_seed(123)
    model = MLP(4, [16, 16], "relu")
    inputs = torch.linspace(-2, 2, 40).reshape(10, 4)
    assert model(inputs).shape == (10, 1)

    set_seed(123)
    first = MLP(4, [16, 16], "relu")
    set_seed(123)
    second = MLP(4, [16, 16], "relu")
    torch.testing.assert_close(first(inputs), second(inputs))


def _tiny_dataset(path: Path) -> Path:
    rng = np.random.default_rng(123)
    arrays = {}
    for split, count in (("train", 24), ("val", 12), ("test", 12)):
        x = rng.normal(size=(count, 4)).astype(np.float32)
        y = (0.2 * x[:, 0] + 0.1).astype(np.float32)
        arrays[f"X_reduced_{split}"] = x
        arrays[f"call_over_K_{split}"] = y
        S = 100.0 + rng.normal(size=count)
        K = np.full(count, 100.0)
        arrays[f"X_{split}"] = np.column_stack([
            S,
            K,
            np.full(count, 1.0),
            np.full(count, 0.05),
            np.full(count, 0.2),
        ])
    train_x = arrays["X_reduced_train"]
    arrays["X_reduced_mean"] = train_x.mean(axis=0)
    arrays["X_reduced_std"] = train_x.std(axis=0)
    np.savez(path, **arrays)
    return path


def test_training_checkpoint_and_evaluation_on_tiny_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _tiny_dataset(tmp_path / "tiny.npz")
    config = tmp_path / "config.yaml"
    config.write_text(
        "nn:\n"
        "  hidden_sizes: [8, 8]\n"
        "  activation: relu\n"
        "  learning_rate: 0.01\n"
        "  batch_size: 8\n"
        "  epochs: 5\n"
        "  seed: 7\n",
        encoding="utf-8",
    )

    result = train_model(
        config,
        dataset,
        "tiny_run",
        hidden_sizes=[8, 8],
        activation="relu",
        learning_rate=0.01,
        batch_size=8,
        epochs=5,
        seed=7,
    )
    assert result["validation_losses"][0] >= result["best_validation_loss"]

    run_dir = tmp_path / "results" / "tiny_run"
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = MLP(checkpoint["input_size"], checkpoint["hidden_sizes"], checkpoint["activation"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with np.load(dataset) as data:
        mean = data["X_reduced_mean"].astype(np.float32)
        std = data["X_reduced_std"].astype(np.float32)
        test_x = ((data["X_reduced_test"] - mean) / std).astype(np.float32)
    with torch.no_grad():
        predictions = model(torch.from_numpy(test_x)).numpy()
    assert predictions.shape == (12, 1)

    checkpoint_predictions = torch.load(
        run_dir / "model.pt", map_location="cpu", weights_only=True
    )
    reloaded = MLP(
        checkpoint_predictions["input_size"],
        checkpoint_predictions["hidden_sizes"],
        checkpoint_predictions["activation"],
    )
    reloaded.load_state_dict(checkpoint_predictions["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reloaded_predictions = reloaded(torch.from_numpy(test_x)).numpy()
    np.testing.assert_array_equal(predictions, reloaded_predictions)

    metrics = evaluate_test(run_dir, dataset)
    for key in (
        "mae_call_over_K",
        "rmse_call_over_K",
        "max_abs_error_call_over_K",
        "mae_price",
        "rmse_price",
        "max_abs_error_price",
        "fraction_price_error_above_0.01",
        "fraction_price_error_above_0.1",
        "true_price_below_0.1",
        "relative_error_price_above_0.1",
    ):
        assert key in metrics
    assert np.isfinite(
        sum([
            metrics["mae_call_over_K"],
            metrics["rmse_call_over_K"],
            metrics["max_abs_error_call_over_K"],
            metrics["mae_price"],
            metrics["rmse_price"],
            metrics["max_abs_error_price"],
        ])
    )
