"""Single source of truth for the project version, read from pyproject.toml."""

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


__version__: str = get_version()
