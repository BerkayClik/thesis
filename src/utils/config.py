"""
Configuration utilities module.

Provides YAML config loading and merging functionality.
"""

import yaml
import os
from typing import Dict, Any


VALID_TARGET_MODES = ("price", "return", "log_return")
VALID_LOSS_TYPES = ("mse", "directional_mse")
# Conservative default: a local lambda sweep on BTC 4h showed corr(pred,true)
# DEGRADES as lambda rises (0.04 -> 0.00), so keep the nudge gentle when a user
# opts into directional loss without specifying lambda_dir.
DEFAULT_LAMBDA_DIR = 0.1
DEFAULT_DIRECTIONAL_K = 1000.0


def resolve_loss(config: Dict[str, Any]):
    """Resolve the training loss selection from a config dict.

    Returns ``(loss_type, lambda_dir, k)``. Default is ``"mse"`` with
    ``lambda_dir=0.0`` so existing configs/runs are unchanged.

    ``directional_mse`` is only valid in return-mode: the sign of a return is a
    direction, but the sign of a price is meaningless, so requesting it in
    price-mode is an error.

    Raises:
        ValueError: unknown loss_type, or directional_mse in price-mode.
    """
    training = config.get("training") or {}
    loss_type = training.get("loss_type", "mse")

    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(
            f"Invalid loss_type {loss_type!r}; expected one of {VALID_LOSS_TYPES}"
        )

    if loss_type == "mse":
        return "mse", 0.0, DEFAULT_DIRECTIONAL_K

    if resolve_target_mode(config) == "price":
        raise ValueError(
            "directional_mse loss requires target_mode 'return' or 'log_return' "
            "(sign of a price is not a direction)"
        )

    lambda_dir = float(training.get("lambda_dir", DEFAULT_LAMBDA_DIR))
    k = float(training.get("directional_k", DEFAULT_DIRECTIONAL_K))
    return "directional_mse", lambda_dir, k


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
