"""Single source of truth for the project version, read from pyproject.toml."""

import subprocess
from pathlib import Path

import tomllib


def get_version() -> str:
    """
    Read the version string from pyproject.toml.

    Falls back to ``"0.0.0-dev"`` if the file cannot be read.

    :return: Version string (e.g., ``"0.1.0"``).
    :rtype: str
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return str(data.get("tool", {}).get("poetry", {}).get("version", "0.0.0-dev"))
    return "0.0.0-dev"


def get_git_commit() -> str:
    """
    Short hash of the checked-out commit, suffixed ``-dirty`` when the tree has changes.

    Recorded by training runs and explanation results so either can be traced to its code.

    :return: e.g. ``"7206375-dirty"``, or ``"unknown"`` outside a git checkout.
    :rtype: str
    """
    try:
        hash_ = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{hash_}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


__version__: str = get_version()
