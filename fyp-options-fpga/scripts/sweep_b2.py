"""Cosine-scheduled ReLU architecture sweep for B2."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optfpga.nn.evaluate import evaluate_test
from optfpga.nn.model import MLP
from optfpga.nn.train import train_model

CONFIGURATIONS = [
    [16],
    [16, 16],
    [32, 32],
    [64, 64],
    [128, 128],
    [32, 32, 32],
    [64, 64, 64],
]
SWEEP_SEEDS = (42, 43, 44)


def _configuration_name(hidden_sizes: list[int]) -> str:
    return "_".join(str(size) for size in hidden_sizes) + "_relu"


def _read_history(run_dir: Path, epochs: int) -> dict[str, Any] | None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(history, dict):
        return None
    validation_losses = history.get("validation_losses")
    if not isinstance(validation_losses, list) or len(validation_losses) != epochs:
        return None
    epoch_numbers = history.get("epochs")
    if epoch_numbers is not None and (
        not isinstance(epoch_numbers, list)
        or len(epoch_numbers) != epochs
        or epoch_numbers[-1] != epochs
    ):
        return None
    return history


def _recompute_saved_run(
    run_dir: Path,
    dataset_path: Path,
) -> tuple[float, float, float, int]:
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = MLP(
        checkpoint["input_size"],
        checkpoint["hidden_sizes"],
        checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with np.load(dataset_path) as data:
        val_inputs = data["X_reduced_val"].astype(np.float32)
        val_targets = data["call_over_K_val"].reshape(-1).astype(np.float64)
        if (run_dir / "normalization.npz").exists():
            with np.load(run_dir / "normalization.npz") as normalization:
                mean = normalization["X_reduced_mean"].astype(np.float32)
                std = normalization["X_reduced_std"].astype(np.float32)
        else:
            mean = data["X_reduced_mean"].astype(np.float32)
            std = data["X_reduced_std"].astype(np.float32)

    normalized_inputs = ((val_inputs - mean) / std).astype(np.float32)
    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized_inputs)).numpy().reshape(-1)
    errors = predictions.astype(np.float64) - val_targets
    return (
        float(np.mean(errors ** 2)),
        float(np.mean(np.abs(errors))),
        float(np.max(np.abs(errors))),
        sum(parameter.numel() for parameter in model.parameters()),
    )


def _resume_run(
    run_dir: Path,
    epochs: int,
    dataset_path: Path,
) -> dict[str, Any] | None:
    history = _read_history(run_dir, epochs)
    checkpoint_path = run_dir / "model.pt"
    if history is None or not checkpoint_path.exists():
        return None

    validation_losses = [float(value) for value in history["validation_losses"]]
    best_epoch = int(np.argmin(validation_losses)) + 1
    summary_path = run_dir / "sweep_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("epochs") == epochs:
                summary = loaded
        except json.JSONDecodeError:
            summary = {}

    required = (
        "final_validation_mse",
        "final_validation_mae",
        "final_validation_max_abs_error",
        "n_parameters",
    )
    if all(key in summary for key in required):
        final_mse = float(summary["final_validation_mse"])
        final_mae = float(summary["final_validation_mae"])
        final_max_error = float(summary["final_validation_max_abs_error"])
        n_parameters = int(summary["n_parameters"])
    else:
        (
            final_mse,
            final_mae,
            final_max_error,
            n_parameters,
        ) = _recompute_saved_run(run_dir, dataset_path)

    training_seconds = float(summary.get("training_seconds", float("nan")))
    return {
        "run_dir": str(run_dir),
        "final_validation_mse": final_mse,
        "final_validation_mae": final_mae,
        "final_validation_max_abs_error": final_max_error,
        "n_parameters": n_parameters,
        "training_seconds": training_seconds,
        "best_validation_epoch": best_epoch,
        "best_validation_epoch_is_final": best_epoch == epochs,
        "resumed": True,
    }


def _save_run_summary(run_dir: Path, run: dict[str, Any], epochs: int) -> None:
    summary = {
        "epochs": epochs,
        "final_validation_mse": run["final_validation_mse"],
        "final_validation_mae": run["final_validation_mae"],
        "final_validation_max_abs_error": run["final_validation_max_abs_error"],
        "n_parameters": run["n_parameters"],
        "training_seconds": run["training_seconds"],
        "best_validation_epoch": run["best_validation_epoch"],
        "best_validation_epoch_is_final": run["best_validation_epoch_is_final"],
    }
    (run_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def _train_seed(
    hidden_sizes: list[int],
    seed: int,
    epochs: int,
    config_path: Path,
    dataset_path: Path,
    no_resume: bool = False,
) -> dict[str, Any]:
    name = _configuration_name(hidden_sizes)
    run_name = f"b2_sweep/{name}_seed_{seed}"
    expected_run_dir = Path("results") / run_name
    if not no_resume:
        resumed = _resume_run(expected_run_dir, epochs, dataset_path)
        if resumed is not None:
            _save_run_summary(expected_run_dir, resumed, epochs)
            resumed.update({"seed": seed, "hidden_sizes": hidden_sizes})
            return resumed

    started = time.perf_counter()
    result = train_model(
        config_path,
        dataset_path,
        run_name,
        hidden_sizes=hidden_sizes,
        activation="relu",
        epochs=epochs,
        seed=seed,
        lr_schedule="cosine",
    )
    elapsed = time.perf_counter() - started
    run_dir = Path(result["run_dir"])
    history = _read_history(run_dir, epochs)
    if history is None:
        raise ValueError(f"Training history is missing or incomplete in {run_dir}")
    best_epoch = int(np.argmin(history["validation_losses"])) + 1
    run = {
        "seed": seed,
        "hidden_sizes": hidden_sizes,
        "run_dir": str(run_dir),
        "n_parameters": result["n_parameters"],
        "final_validation_mse": result["final_validation_mse"],
        "final_validation_mae": result["final_validation_mae"],
        "final_validation_max_abs_error": result["final_validation_max_abs_error"],
        "training_seconds": elapsed,
        "best_validation_epoch": best_epoch,
        "best_validation_epoch_is_final": best_epoch == epochs,
        "resumed": False,
    }
    _save_run_summary(run_dir, run, epochs)
    return run


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _aggregate(
    hidden_sizes: list[int],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    def values(field: str) -> list[float]:
        return [float(run[field]) for run in runs]

    return {
        "hidden_sizes": hidden_sizes,
        "activation": "relu",
        "n_parameters": int(runs[0]["n_parameters"]),
        "seeds": [int(run["seed"]) for run in runs],
        "mean_final_validation_mse": float(np.mean(values("final_validation_mse"))),
        "std_final_validation_mse": _sample_std(values("final_validation_mse")),
        "mean_validation_mae": float(np.mean(values("final_validation_mae"))),
        "std_validation_mae": _sample_std(values("final_validation_mae")),
        "mean_validation_max_abs_error": float(
            np.mean(values("final_validation_max_abs_error"))
        ),
        "std_validation_max_abs_error": _sample_std(
            values("final_validation_max_abs_error")
        ),
        "mean_training_seconds": float(np.mean(values("training_seconds"))),
        "std_training_seconds": _sample_std(values("training_seconds")),
        "best_validation_epoch_is_final_by_seed": [
            {
                "seed": int(run["seed"]),
                "best_validation_epoch": int(run["best_validation_epoch"]),
                "is_final_epoch": bool(run["best_validation_epoch_is_final"]),
            }
            for run in runs
        ],
        "runs": runs,
    }


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "hidden_sizes",
        "activation",
        "n_parameters",
        "mean_final_validation_mse",
        "std_final_validation_mse",
        "mean_validation_mae",
        "std_validation_mae",
        "mean_validation_max_abs_error",
        "std_validation_max_abs_error",
        "seeds_used",
        "best_validation_epoch_is_final_by_seed",
        "mean_training_seconds",
        "std_training_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "hidden_sizes": json.dumps(row["hidden_sizes"]),
                "activation": row["activation"],
                "n_parameters": row["n_parameters"],
                "mean_final_validation_mse": row["mean_final_validation_mse"],
                "std_final_validation_mse": row["std_final_validation_mse"],
                "mean_validation_mae": row["mean_validation_mae"],
                "std_validation_mae": row["std_validation_mae"],
                "mean_validation_max_abs_error": row["mean_validation_max_abs_error"],
                "std_validation_max_abs_error": row["std_validation_max_abs_error"],
                "seeds_used": json.dumps(row["seeds"]),
                "best_validation_epoch_is_final_by_seed": json.dumps(
                    row["best_validation_epoch_is_final_by_seed"]
                ),
                "mean_training_seconds": row["mean_training_seconds"],
                "std_training_seconds": row["std_training_seconds"],
            })


def _aggregate_runs(completed_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for run in completed_runs:
        key = tuple(run["hidden_sizes"])
        grouped.setdefault(key, []).append(run)
    rows = [
        _aggregate(list(hidden_sizes), runs)
        for hidden_sizes, runs in grouped.items()
    ]
    return sorted(
        rows,
        key=lambda row: (row["n_parameters"], tuple(row["hidden_sizes"])),
    )


def _select_configuration(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], float, float, list[dict[str, Any]]]:
    if not rows:
        raise ValueError("at least one completed configuration is required")
    best = min(rows, key=lambda row: row["mean_final_validation_mse"])
    sample_variances = [
        float(row["std_final_validation_mse"]) ** 2
        for row in rows
        if len(row["seeds"]) >= 2
        and np.isfinite(row["std_final_validation_mse"])
    ]
    pooled_std = float(np.sqrt(np.mean(sample_variances))) if sample_variances else 0.0
    threshold = float(best["mean_final_validation_mse"] + 2.0 * pooled_std)
    eligible = [
        row for row in rows
        if row["mean_final_validation_mse"] <= threshold
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["n_parameters"],
            row["mean_final_validation_mse"],
        ),
    )
    return selected, best, pooled_std, threshold, eligible


def run_sweep(
    dataset_path: Path,
    config_path: Path,
    epochs: int = 300,
    seeds: int = 2,
    no_resume: bool = False,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if seeds not in (1, 2, 3):
        raise ValueError("seeds must be 1, 2, or 3")

    output_dir = Path("results") / "b2_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "validation_metrics.csv"
    completed_runs: list[dict[str, Any]] = []
    runs_done = 0
    total_runs = len(CONFIGURATIONS) * seeds
    if seeds == 2:
        total_runs += 3

    def complete_run(hidden_sizes: list[int], seed: int) -> dict[str, Any]:
        nonlocal runs_done
        run = _train_seed(
            hidden_sizes,
            seed,
            epochs,
            config_path,
            dataset_path,
            no_resume=no_resume,
        )
        completed_runs.append(run)
        runs_done += 1
        _write_summary_csv(_aggregate_runs(completed_runs), csv_path)
        print(
            f"Progress [{runs_done}/{total_runs}]: "
            f"configuration={hidden_sizes} seed={seed} "
            f"final_validation_mse={run['final_validation_mse']:.10g} "
            f"seconds={run['training_seconds']:.2f} "
            f"status={'RESUMED' if run['resumed'] else 'TRAINED'}"
        )
        if not run["best_validation_epoch_is_final"]:
            print(
                f"WARNING: configuration={hidden_sizes} seed={seed} "
                f"best-validation epoch {run['best_validation_epoch']} "
                f"does not equal final epoch {epochs}."
            )
        return run

    initial_seeds = SWEEP_SEEDS[:seeds]
    runs_by_configuration: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for hidden_sizes in CONFIGURATIONS:
        key = tuple(hidden_sizes)
        runs_by_configuration[key] = [
            complete_run(hidden_sizes, seed)
            for seed in initial_seeds
        ]

    if seeds == 2:
        preliminary = _aggregate_runs([
            run
            for runs in runs_by_configuration.values()
            for run in runs
        ])
        top_three = sorted(
            preliminary,
            key=lambda row: row["mean_final_validation_mse"],
        )[:3]
        print(
            "Adding a third seed (44) to: "
            + ", ".join(str(row["hidden_sizes"]) for row in top_three)
        )
        for row in top_three:
            key = tuple(row["hidden_sizes"])
            third_run = complete_run(list(key), SWEEP_SEEDS[2])
            runs_by_configuration[key].append(third_run)

    rows = _aggregate_runs(completed_runs)
    _write_summary_csv(rows, csv_path)
    selected, best, pooled_std, threshold, eligible = _select_configuration(rows)
    eligible_text = ", ".join(
        f"{row['hidden_sizes']} (mean={row['mean_final_validation_mse']:.10g})"
        for row in eligible
    )
    print(f"Best mean final-epoch validation MSE: {best['mean_final_validation_mse']:.10g}")
    print(f"Pooled standard deviation: {pooled_std:.10g}")
    print(f"Selection threshold: {threshold:.10g}")
    print(f"Eligible configurations: {eligible_text}")
    print(
        f"Selected hidden_sizes={selected['hidden_sizes']}, activation=relu "
        f"({selected['n_parameters']} parameters). It is the smallest network "
        f"whose mean final-epoch validation MSE "
        f"({selected['mean_final_validation_mse']:.10g}) is no greater than "
        f"the best mean plus two pooled standard deviations."
    )

    selected_test_metrics = []
    for run in selected["runs"]:
        run_dir = Path(run["run_dir"])
        metrics = evaluate_test(run_dir, dataset_path)
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        selected_test_metrics.append({
            "seed": run["seed"],
            "metrics": metrics,
        })

    test_metrics_path = output_dir / "selected_test_metrics.json"
    test_metrics_path.write_text(
        json.dumps({
            "hidden_sizes": selected["hidden_sizes"],
            "activation": "relu",
            "selected_by": "smallest parameter count within 2 pooled standard deviations of mean final validation MSE",
            "seeds": selected_test_metrics,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "best": best,
        "pooled_std": pooled_std,
        "threshold": threshold,
        "eligible": eligible,
        "selected": selected,
        "validation_csv": csv_path,
        "test_metrics_json": test_metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/black_scholes_dataset.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--seeds",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Seeds per configuration; with 2 seeds the top three receive a third (default: 2).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Retrain runs even when a complete run already exists.",
    )
    args = parser.parse_args()
    result = run_sweep(
        args.dataset,
        args.config,
        args.epochs,
        args.seeds,
        args.no_resume,
    )
    print(f"Validation summary saved to {result['validation_csv']}")
    print(f"Selected test metrics saved to {result['test_metrics_json']}")


if __name__ == "__main__":
    main()
