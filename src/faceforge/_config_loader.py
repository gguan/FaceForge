"""Project config loader.

Reads config.yaml from the project root and merges it with dataclass defaults.
Sections map directly to Stage1Config / Stage2Config field names.

Usage::

    from faceforge._config_loader import load_stage1_overrides, load_stage2_overrides

    overrides = load_stage2_overrides()           # from default config.yaml
    overrides = load_stage2_overrides('my.yaml')  # from custom path

    config = Stage2Config(**{**overrides, 'device': 'cuda:0'})
"""

from pathlib import Path
from typing import Any

import yaml

from faceforge._paths import PROJECT_ROOT

_DEFAULT_CONFIG = PROJECT_ROOT / 'config.yaml'


def _load_yaml(config_path: Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        return {}
    with open(path, encoding='utf-8-sig') as f:
        return yaml.safe_load(f) or {}


def load_stage1_overrides(config_path: Path | None = None) -> dict[str, Any]:
    """Return stage1 section of the config file as a flat dict."""
    return _load_yaml(config_path).get('stage1', {})


def load_stage2_overrides(config_path: Path | None = None) -> dict[str, Any]:
    """Return stage2 section as a flat dict, with lr_decay_milestones converted to tuple."""
    data = _load_yaml(config_path).get('stage2', {})
    if 'lr_decay_milestones' in data:
        data['lr_decay_milestones'] = tuple(data['lr_decay_milestones'])
    return data
