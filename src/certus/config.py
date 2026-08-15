"""Central Hydra entry point for resolving configs/ into a single OmegaConf
tree.

Every experiment run (calibration, LOSO, figure generation) should load its
configuration through `load_config` rather than reading YAML files
directly, so that config composition (data + score + conformal +
experiment) stays consistent and every run can log the resolved config
verbatim (roadmap Section 4.5 -- provenance).
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose a config by name (e.g. 'experiment/e1_validity') with optional overrides."""
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return hydra.compose(config_name=config_name, overrides=overrides or [])


if __name__ == "__main__":
    cfg = load_config("experiment/e1_validity")
    print(OmegaConf.to_yaml(cfg))
