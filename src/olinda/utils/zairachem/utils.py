"""Utilities."""

import os
import shutil
import glob
from typing import Any, Callable
from olinda.utils.zairachem.zairachem import ZairaChemPredictor
   
def run_zairachem(model_path: str) -> Callable:
    """Utility function to run ZairaChem model predictions.

    Args:
        model_path (str): Path to ZairaChem model.

    Returns:
        Callable: Util function.
    """

    def execute(smiles_path: str) -> list:
        folds_exist = [file for file in glob.glob(os.path.join(model_path, "distill", "*")) if "fold" in file]
        model_output = os.path.join(model_path, "distill", "fold" + str(len(folds_exist)+1))

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

