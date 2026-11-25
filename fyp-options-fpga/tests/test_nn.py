import json
import sys
from pathlib import Path

import numpy as np
import torch

from optfpga.nn.evaluate import evaluate_test
from optfpga.nn.model import MLP
import optfpga.nn.train as train_module
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


def test_training_cli_resolves_default_paths_from_project_root(monkeypatch) -> None:
    captured = {}

    def fake_train_model(config_path, dataset_path, run_name, **kwargs):
        captured.update({
            "config_path": config_path,
            "dataset_path": dataset_path,
            "run_name": run_name,
            "kwargs": kwargs,
        })
        return {
            "best_validation_loss": 0.1,
            "train_losses": [0.2],
            "validation_losses": [0.1],
            "best_validation_epoch": 1,
            "run_dir": run_name,
        }

    monkeypatch.setattr(train_module, "train_model", fake_train_model)
    monkeypatch.setattr(sys, "argv", ["train.py"])

    train_module.main()

    assert captured["config_path"] is None
    assert captured["dataset_path"] == train_module.PROJECT_ROOT / "data" / "black_scholes_dataset.npz"
    assert captured["run_name"] == str(train_module.PROJECT_ROOT / "results" / "b2_baseline")


def test_training_learns_fixed_linear_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(321)
    train_x = rng.normal(size=(200, 4)).astype(np.float32)
    val_x = rng.normal(size=(40, 4)).astype(np.float32)
    train_y = (0.3 * train_x[:, 0] - 0.2 * train_x[:, 1] + 0.1 * train_x[:, 2] + 0.05).astype(np.float32)
    val_y = (0.3 * val_x[:, 0] - 0.2 * val_x[:, 1] + 0.1 * val_x[:, 2] + 0.05).astype(np.float32)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    np.savez(
        tmp_path / "linear.npz",
        X_reduced_train=train_x,
        X_reduced_val=val_x,
        call_over_K_train=train_y,
        call_over_K_val=val_y,
        X_reduced_mean=mean,
        X_reduced_std=std,
    )

    config = tmp_path / "linear_config.yaml"
    config.write_text(
        "nn:\n"
        "  hidden_sizes: [16]\n"
        "  activation: relu\n"
        "  learning_rate: 0.01\n"
        "  batch_size: 32\n"
        "  epochs: 30\n"
        "  seed: 9\n",
        encoding="utf-8",
    )
    result = train_model(config, tmp_path / "linear.npz", "linear_run", epochs=30)
    assert result["n_parameters"] > 0
    assert result["train_losses"][-1] < 0.5 * result["train_losses"][0]


def test_train_model_never_requires_test_arrays(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(456)
    train_x = rng.normal(size=(30, 4)).astype(np.float32)
    val_x = rng.normal(size=(10, 4)).astype(np.float32)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    np.savez(
        tmp_path / "no_test.npz",
        X_reduced_train=train_x,
        X_reduced_val=val_x,
        call_over_K_train=np.zeros(30, dtype=np.float32),
        call_over_K_val=np.zeros(10, dtype=np.float32),
        X_reduced_mean=mean,
        X_reduced_std=std,
    )
    config = tmp_path / "no_test_config.yaml"
    config.write_text(
        "nn:\n"
        "  hidden_sizes: [4]\n"
        "  activation: tanh\n"
        "  learning_rate: 0.01\n"
        "  batch_size: 10\n"
        "  epochs: 2\n"
        "  seed: 11\n",
        encoding="utf-8",
    )
    result = train_model(config, tmp_path / "no_test.npz", "no_test_run", epochs=2)
    assert (tmp_path / "results" / "no_test_run" / "model.pt").exists()
    assert result["n_parameters"] > 0


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
    run_dir = tmp_path / "results" / "tiny_run"
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert history["epochs"] == [1, 2, 3, 4, 5]
    assert history["train_losses"] == result["train_losses"]
    assert history["validation_losses"] == result["validation_losses"]
    assert result["final_validation_mse"] == result["validation_losses"][-1]
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

    with np.load(run_dir / "weights.npz") as weights:
        numpy_predictions = test_x
        for index in range(len(checkpoint["hidden_sizes"])):
            numpy_predictions = numpy_predictions @ weights[f"weight_{index}"].T
            numpy_predictions = numpy_predictions + weights[f"bias_{index}"]
            if checkpoint["activation"] == "relu":
                numpy_predictions = np.maximum(numpy_predictions, 0.0)
            else:
                numpy_predictions = np.tanh(numpy_predictions)
        final_index = len(checkpoint["hidden_sizes"])
        numpy_predictions = numpy_predictions @ weights[f"weight_{final_index}"].T
        numpy_predictions = numpy_predictions + weights[f"bias_{final_index}"]
    np.testing.assert_allclose(numpy_predictions, predictions, rtol=1e-5, atol=1e-5)

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


def test_evaluate_reports_nan_for_empty_moneyness_bin(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(789)
    train_x = rng.normal(size=(20, 4)).astype(np.float32)
    val_x = rng.normal(size=(8, 4)).astype(np.float32)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    np.savez(
        tmp_path / "empty_bin.npz",
        X_reduced_train=train_x,
        X_reduced_val=val_x,
        call_over_K_train=np.zeros(20, dtype=np.float32),
        call_over_K_val=np.zeros(8, dtype=np.float32),
        X_reduced_mean=mean,
        X_reduced_std=std,
        X_reduced_test=np.ones((4, 4), dtype=np.float32),
        call_over_K_test=np.full(4, 0.1, dtype=np.float32),
        X_test=np.column_stack([
            np.full(4, 100.0),
            np.full(4, 100.0),
            np.full(4, 1.0),
            np.full(4, 0.05),
            np.full(4, 0.2),
        ]),
    )
    config = tmp_path / "empty_bin_config.yaml"
    config.write_text(
        "nn:\n"
        "  hidden_sizes: [4]\n"
        "  activation: relu\n"
        "  learning_rate: 0.01\n"
        "  batch_size: 10\n"
        "  epochs: 2\n"
        "  seed: 12\n",
        encoding="utf-8",
    )
    train_model(config, tmp_path / "empty_bin.npz", "empty_bin_run", epochs=2)

    metrics = evaluate_test(tmp_path / "results" / "empty_bin_run", tmp_path / "empty_bin.npz")
    for name in ("below_0.9", "above_1.1"):
        call_metrics = metrics["moneyness_bins"][name]["call_over_K"]
        price_metrics = metrics["moneyness_bins"][name]["price"]
        assert call_metrics["count"] == 0
        assert price_metrics["count"] == 0
        assert np.isnan(call_metrics["mae"])
        assert np.isnan(call_metrics["rmse"])
        assert np.isnan(call_metrics["max_abs_error"])
        assert np.isnan(price_metrics["mae"])
        assert np.isnan(price_metrics["rmse"])
        assert np.isnan(price_metrics["max_abs_error"])
