"""Utilities."""

import os
import glob
import pandas as pd
from typing import Any, Callable
from olinda.utils.zairachem.zairachem import ZairaChemPredictor
from olinda.configs.vars import REF_FOLD_SIZE
   
def run_zairachem(model_path: str) -> Callable:
    """Utility function to run ZairaChem model predictions.

    Args:
        model_path (str): Path to ZairaChem model.

    Returns:
        Callable: Util function.
    """

    def execute(smiles_path: str) -> list:
        #Check if some folds have already been processed and calculate the next fold
        folds_exist = [file for file in glob.glob(os.path.join(model_path, "distill", "*")) if "fold" in file]
        curr_fold = len(folds_exist)+1
        model_output_path = os.path.join(model_path, "distill", "fold" + str(curr_fold))
        os.makedirs(model_output_path, exist_ok=True)
        subset_smiles_path = os.path.join(model_output_path, "reference_library_subset.csv")
        df = pd.read_csv(smiles_path)
        subset_df = df.iloc[curr_fold*REF_FOLD_SIZE : (curr_fold+1)*REF_FOLD_SIZE]
        subset_df.to_csv(subset_smiles_path, index=False)

        zp = ZairaChemPredictor(subset_smiles_path, model_path, model_output_path)
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

