"""Project configuration loading."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load project settings from a YAML file.

    By default, this loads ``configs/default.yaml`` relative to the project root.
    """
    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    )
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    return config
