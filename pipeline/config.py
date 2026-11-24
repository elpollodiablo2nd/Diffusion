"""Settings loading and validation for the de novo binder pipeline.

All relative paths in settings.json are resolved against the directory that
contains the settings file (the repo root by default).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Settings:
    def __init__(self, raw: dict, root: Path):
        self.raw = raw
        self.root = root

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"Missing required settings key: {dotted!r}")
        return value

    def path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def device(self) -> str:
        return self.get("device", "gpu")

    @property
    def seed(self) -> int:
        return int(self.get("seed", 0))

    @property
    def target_pdb(self) -> Path:
        return self.path(self.require("paths.target_pdb"))

    @property
    def workdir(self) -> Path:
        return self.path(self.require("paths.workdir"))

    def stage_dir(self, name: str) -> Path:
        return self.workdir / name


def load_settings(
    settings_path: str | Path,
    device: Optional[str] = None,
    smoke: bool = False,
    workdir: Optional[str | Path] = None,
) -> Settings:
    settings_path = Path(settings_path).resolve()
    root = settings_path.parent
    with settings_path.open() as fh:
        raw = json.load(fh)
    if smoke:
        smoke_overrides = raw.pop("smoke", {})
        _deep_merge(raw, copy.deepcopy(smoke_overrides))
    if device is not None:
        if device not in ("gpu", "cpu"):
            raise ValueError(f"device must be 'gpu' or 'cpu', got {device!r}")
        raw["device"] = device
    if workdir is not None:
        raw.setdefault("paths", {})["workdir"] = str(workdir)
    settings = Settings(raw, root)
    if not settings.target_pdb.exists():
        raise FileNotFoundError(f"Target PDB not found: {settings.target_pdb}")
    return settings
