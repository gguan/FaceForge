"""Project path resolution utilities.

Locates the project root by searching upward for a sentinel file
(pyproject.toml or .git directory), so paths remain valid regardless
of the depth at which a module sits in the source tree.
"""

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk upward from *start* until a sentinel is found.

    Sentinels checked (in order):
      - ``pyproject.toml``   — standard Python project marker
      - ``.git``             — git repository root

    Args:
        start: Directory to begin the search (inclusive).

    Returns:
        Absolute :class:`~pathlib.Path` to the project root.

    Raises:
        RuntimeError: If neither sentinel is found before the filesystem root.
    """
    for directory in [start.resolve(), *start.resolve().parents]:
        if (directory / 'pyproject.toml').exists():
            return directory
        if (directory / '.git').exists():
            return directory
    raise RuntimeError(
        f"Project root not found: no 'pyproject.toml' or '.git' directory "
        f"found when searching upward from '{start}'. "
        "Ensure the package is installed from within the FaceForge repository."
    )


#: Absolute path to the repository root, resolved once at import time.
PROJECT_ROOT: Path = _find_project_root(Path(__file__).parent)
