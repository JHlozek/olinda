"""Utilities."""

from io import BufferedWriter
import os
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional

from cbor2 import CBORDecoder
import gin
from xdg import xdg_data_home

import onnxruntime as rt


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


def run_onnx_runtime(onnx_model: Any) -> Callable:
    """Utility function to execute ONNX runtime.

    Args:
        onnx_model (str): ONNX model object

    Returns:
        Callable: Util function.
    """
    
    def execute(x: list) -> list:
        onnx_rt = rt.InferenceSession(onnx_model.SerializeToString())
        output_names = [n.name for n in onnx_model.graph.output]
        
        #adapt for single versus batched queries
        if np.array(x).shape[0] == 1:
            preds = onnx_rt.run(output_names, {"input": x})[0]
        else:
            preds = [onnx_rt.run(output_names, {"input": val})[0] for val in x]
            
        #flatten list
        preds = [float(ele) for ele in preds]
        return preds
    return execute
