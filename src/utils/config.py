"""
Configuration utilities module.

Provides YAML config loading and merging functionality.
"""

import yaml
import os
from typing import Dict, Any


VALID_TARGET_MODES = ("price", "return", "log_return")


def resolve_target_mode(config: Dict[str, Any]) -> str:
    """Resolve the prediction target mode from a config dict.

    The flag is additive and non-destructive: when absent it defaults to
    ``"price"`` so every existing config keeps its current behavior.

    Lookup order: ``config["data"]["target_mode"]`` then top-level
    ``config["target_mode"]``.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        One of ``"price"``, ``"return"``, ``"log_return"``.

    Raises:
        ValueError: If an unknown target_mode value is provided.
    """
    data = config.get("data") or {}
    mode = data.get("target_mode", config.get("target_mode", "price"))

    if mode not in VALID_TARGET_MODES:
        raise ValueError(
            f"Invalid target_mode {mode!r}; expected one of {VALID_TARGET_MODES}"
        )
    return mode


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Override values take precedence over base values.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_config(
    base_path: str = "configs/base.yaml",
    experiment_path: str = None
) -> Dict[str, Any]:
    """
    Load and merge base and experiment configurations.

    Args:
        base_path: Path to base config file.
        experiment_path: Optional path to experiment config file.

    Returns:
        Final merged configuration.
    """
    config = load_config(base_path)

    if experiment_path:
        exp_config = load_config(experiment_path)
        config = merge_configs(config, exp_config)

    return config
