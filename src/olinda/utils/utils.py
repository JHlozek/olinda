"""Utilities."""

from io import BufferedWriter
import os
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional

from cbor2 import CBORDecoder
import gin
from xdg import xdg_data_home


def get_package_root_path() -> Path:
    """Get path of the package root.

    Returns:
        Path: Package root path
    """
    return Path(__file__).parents[0].absolute()


@gin.configurable
def get_workspace_path(override: Optional[Path] = None) -> Path:
    """Get path of the Olinda workspace.

    Args:
        override (Optional[Path], optional): _description_. Defaults to None.

    Returns:
        Path: Package root path
    """
    if override is None:
        workspace_path = xdg_data_home() / "olinda" / "workspace"
    else:
        workspace_path = Path(override)
    # Create dir if not already present
    os.makedirs(workspace_path, exist_ok=True)
    os.makedirs(workspace_path / "reference", exist_ok=True)
    return workspace_path


def calculate_cbor_size(fp: BufferedWriter) -> int:
    """Calculate no of processed instances in a cbor file."""
    decoder = CBORDecoder(fp)
    size = 0
    while fp.peek(1):
        decoder.decode()
        size += 1
    return size

