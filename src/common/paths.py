from pathlib import Path
import sys


def get_project_root() -> Path:
    """Return the project root identified by pyproject.toml."""
    path = Path.cwd()

    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise FileNotFoundError("Project root not found.")


def add_project_root_to_path() -> Path:
    """Add the project root to sys.path and return it."""
    root = get_project_root()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root