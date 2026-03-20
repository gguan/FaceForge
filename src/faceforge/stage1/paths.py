"""Shared path and compatibility helpers for Stage 1 integrations."""

from __future__ import annotations

import filecmp
import inspect
import os
import shutil
from pathlib import Path

import numpy as np

from faceforge._paths import PROJECT_ROOT


def get_project_root() -> str:
    """Return the repository root as an absolute string path."""
    return str(PROJECT_ROOT)


def _obj_has_uv_coordinates(path: Path) -> bool:
    """Check whether an OBJ file contains UV coordinate records."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return any(line.startswith("vt ") for line in fh)
    except OSError:
        return False


def get_deca_topology_path() -> str:
    """Return a topology OBJ that includes UV coordinates for DECA."""
    candidates = [
        PROJECT_ROOT / "data" / "pretrained" / "head_template.obj",
        PROJECT_ROOT / "data" / "pretrained" / "FLAME2020" / "head_template.obj",
        PROJECT_ROOT / "submodules" / "pixel3dmm" / "assets" / "head_template.obj",
    ]
    for candidate in candidates:
        if candidate.exists() and _obj_has_uv_coordinates(candidate):
            return str(candidate)

    fallback = candidates[0]
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError("Could not find a DECA topology OBJ file.")


def ensure_symlink_or_copy(source: str, target: str) -> None:
    """Create *target* as a symlink to *source*, falling back to a copy."""
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copy2(source, target)


def ensure_file_matches(source: str, target: str) -> None:
    """Ensure *target* exists and matches *source*."""
    if os.path.lexists(target):
        try:
            if os.path.samefile(source, target):
                return
        except OSError:
            pass

        if os.path.isfile(target) and filecmp.cmp(source, target, shallow=False):
            return

        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.unlink(target)

    ensure_symlink_or_copy(source, target)


def ensure_inspect_getargspec() -> None:
    """Backfill the removed inspect.getargspec API for older deps."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def ensure_numpy_legacy_aliases() -> None:
    """Backfill NumPy aliases removed in newer releases."""
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
