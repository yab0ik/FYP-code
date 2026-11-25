import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import sweep_b2


def test_stale_history_retrains_and_completed_history_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("results/b2_sweep/16_relu_seed_42")
    run_dir.mkdir(parents=True)
    (run_dir / "history.json").write_text(
        json.dumps({
            "epochs": [1, 2],
            "train_losses": [0.2, 0.1],
            "validation_losses": [0.3, 0.2],
        }),
        encoding="utf-8",
    )
    (run_dir / "model.pt").write_bytes(b"stale checkpoint")
    calls = []

    def fake_train_model(config_path, dataset_path, run_name, **kwargs):
        calls.append((config_path, dataset_path, run_name, kwargs))
        target = Path("results") / run_name
        target.mkdir(parents=True, exist_ok=True)
        validation_losses = [0.3] * 299 + [0.1]
        (target / "history.json").write_text(
            json.dumps({
                "epochs": list(range(1, 301)),
                "train_losses": [0.2] * 300,
                "validation_losses": validation_losses,
            }),
            encoding="utf-8",
        )
        (target / "model.pt").write_bytes(b"completed checkpoint")
        return {
            "run_dir": target,
            "n_parameters": 97,
            "final_validation_mse": 0.1,
            "final_validation_mae": 0.2,
            "final_validation_max_abs_error": 0.3,
        }

    monkeypatch.setattr(sweep_b2, "train_model", fake_train_model)
    first = sweep_b2._train_seed(
        [16],
        42,
        300,
        Path("config.yaml"),
        Path("dataset.npz"),
    )

    assert len(calls) == 1
    assert first["resumed"] is False
    assert first["final_validation_mse"] == 0.1
    assert first["best_validation_epoch"] == 300

    monkeypatch.setattr(
        sweep_b2,
        "train_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected retraining")),
    )
    second = sweep_b2._train_seed(
        [16],
        42,
        300,
        Path("config.yaml"),
        Path("dataset.npz"),
    )
    assert second["resumed"] is True
    assert second["final_validation_mse"] == 0.1
    assert second["training_seconds"] == first["training_seconds"]


def test_selection_uses_pooled_sample_standard_deviation() -> None:
    rows = [
        {
            "hidden_sizes": [64, 64],
            "n_parameters": 100,
            "mean_final_validation_mse": 1.0,
            "std_final_validation_mse": 0.5,
            "seeds": [42, 43],
        },
        {
            "hidden_sizes": [16],
            "n_parameters": 20,
            "mean_final_validation_mse": 1.98,
            "std_final_validation_mse": 0.6,
            "seeds": [42, 43],
        },
        {
            "hidden_sizes": [8],
            "n_parameters": 10,
            "mean_final_validation_mse": 2.0,
            "std_final_validation_mse": 0.35,
            "seeds": [42, 43],
        },
    ]

    selected, best, pooled_std, threshold, eligible = sweep_b2._select_configuration(rows)

    expected_pooled_std = np.sqrt((0.5 ** 2 + 0.6 ** 2 + 0.35 ** 2) / 3)
    np.testing.assert_allclose(pooled_std, expected_pooled_std)
    np.testing.assert_allclose(threshold, 1.0 + 2.0 * expected_pooled_std)
    assert best["hidden_sizes"] == [64, 64]
    assert [row["hidden_sizes"] for row in eligible] == [[64, 64], [16]]
    assert selected["hidden_sizes"] == [16]
