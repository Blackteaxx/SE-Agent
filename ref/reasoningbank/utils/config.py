"""Configuration management for the ReasoningBank library."""

from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration settings.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)
