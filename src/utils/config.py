"""Configuration loading utilities."""

from typing import Dict

import yaml


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
