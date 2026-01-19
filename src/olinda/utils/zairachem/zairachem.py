"""ZairaChem predictor"""

import pandas as pd
import os
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import warnings
import logging
from loguru import logger
import subprocess

class ZairaChemPredictor(object):
    def __init__(self, input_file, model_dir, output_dir):
        self.input_file = input_file
        self.model_dir = model_dir
        self.output_dir = output_dir
    
    def predict(self):
        with HiddenPrints():
            subprocess.run(["conda", "run", "-n", "zairachem", "zairachem", "predict", "-i", self.input_file, "-m", self.model_dir, "-o", self.output_dir])            
        return self.clean_output(self.output_dir)
   
    def clean_output(self, path):
        results = pd.read_csv(os.path.join(path, "output_table.csv"))
        col_names = results.columns.values.tolist()
        
        results.rename({'pred-value': 'pred'}, axis=1, inplace=True)
        results.rename({"true-value": 'true'}, axis=1, inplace=True)
        return results[["smiles", 'pred', 'true']]
        
@contextmanager
def HiddenPrints():
    """A context manager that redirects stdout and stderr to devnull"""
    
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            with warnings.catch_warnings():
                logger.disable("zairachem")
                warnings.simplefilter('ignore')
            try:
                yield (err, out)
            finally:
                warnings.simplefilter('default')
