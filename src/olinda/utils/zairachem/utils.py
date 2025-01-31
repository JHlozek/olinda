"""Utilities."""

import os
import shutil
from typing import Any, Callable
from olinda.utils.utils import get_workspace_path
from olinda.utils.zairachem.zairachem import ZairaChemPredictor
   
def run_zairachem(model_path: str) -> Callable:
    """Utility function to run ZairaChem model predictions.

    Args:
        model_path (str): Path to ZairaChem model.

    Returns:
        Callable: Util function.
    """

    def execute(smiles_path: str) -> list:
        model_output = os.path.join(get_workspace_path(), "zairachem_output_dir")
        if os.path.exists(model_output):
            shutil.rmtree(model_output)
    
        zp = ZairaChemPredictor(smiles_path, model_path, model_output)
        return zp.predict()
    return execute    

def get_zairachem_training_preds(model_path: str) -> Callable:
    """Utility function to return the training set predictions of a ZairaChem model.

    Args:
        model_path (str): Path to ZairaChem model.

    Returns:
        Callable: Util function.
    """

    def execute() -> Any:  
        zp = ZairaChemPredictor("", model_path, "")
        return zp.clean_output(model_path)
    return execute   

