"""Configuration management for the GlobalMemory module."""

from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Loads configuration from a YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
