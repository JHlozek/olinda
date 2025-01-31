"""ZairaChem predictor"""
ZAIRACHEM_PATH = "/home/Jason/code/zairachem-docker"

#import zairachem
import ersilia
from ersilia import ErsiliaModel

import pandas as pd
import os
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import shutil
import json
import sys
import warnings
import logging
from loguru import logger
import glob
from pathlib import Path
from progress.bar import Bar
import subprocess

class ZairaChemPredictor(object):
    def __init__(self, input_file, model_dir, output_dir):
        self.input_file = input_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.precalc_path = os.path.dirname(self.input_file)
    
    def predict(self):
        with Bar("ZairaChem Stage", max=7) as bar:
            with HiddenPrints():
                subprocess.run(["conda", "run", "-n", "zairasetup", "python", os.path.join(ZAIRACHEM_PATH, "01_setup/zairasetup/run_predict.py"), "-i", self.input_file, "-m", self.model_dir, "-o", self.output_dir])
                self.data_files()
                bar.next()

                self.precalc_descriptors()
                subprocess.run(["conda", "run", "-n", "zairadescribe", "python", os.path.join(ZAIRACHEM_PATH, "02_describe/zairadescribe/run.py")])
                bar.next()
               
                subprocess.run(["conda", "run", "-n", "zairatreat", "python", os.path.join(ZAIRACHEM_PATH, "03_treat/zairatreat/run.py")])
                bar.next()
            
                subprocess.run(["conda", "run", "-n", "zairaestimate", "python", os.path.join(ZAIRACHEM_PATH, "04_estimate/zairaestimate/run.py")])
                bar.next()
               
                subprocess.run(["conda", "run", "-n", "zairapool", "python", os.path.join(ZAIRACHEM_PATH, "05_pool/zairapool/run.py")])
                bar.next()
               
                subprocess.run(["conda", "run", "-n", "zairareport", "python", os.path.join(ZAIRACHEM_PATH, "06_report/zairareport/run.py")])
                bar.next()

                subprocess.run(["conda", "run", "-n", "zairafinish", "python", os.path.join(ZAIRACHEM_PATH, "07_finish/zairafinish/run.py")])
                bar.next()
            
        return self.clean_output(self.output_dir)

    def data_files(self):
        #update mapping file
        shutil.copy(os.path.join(self.precalc_path, "data", "mapping.csv"), os.path.join(self.output_dir, "data"))
        shutil.copy(os.path.join(self.precalc_path, "data", "data.csv"), os.path.join(self.output_dir, "data"))
 
    def precalc_descriptors(self) -> None:
        precalc_descs = [os.path.basename(desc_path) for desc_path in list(glob.glob(os.path.join(self.precalc_path, "descriptors", "*")))]        
        
        done = []
        with open(os.path.join(self.model_dir, "descriptors", "done_eos.json"), "r") as calculated_desc_file:
            desc_list = json.load(calculated_desc_file)
            for desc in desc_list:
                if desc in precalc_descs:
                    shutil.copytree(os.path.join(self.precalc_path, "descriptors", desc), os.path.join(self.output_dir, "descriptors", desc))
                    done.append(desc)
                else:
                    #make folder and run ersilia model
                    with ErsiliaModel(desc) as em_api:
                        os.makedirs(os.path.join(self.output_dir, "descriptors", desc))
                        em_api.api(input=self.input_file, output=os.path.join(self.output_dir, "descriptors", desc, "raw.h5"))
                        done.append(desc)

        #update json descriptor file
        with open(os.path.join(self.output_dir, "descriptors", "done_eos.json"), "w") as done_file:
            json.dump(done, done_file)

    
    def clean_output(self, path):
        results = pd.read_csv(os.path.join(path, "output_table.csv"))
        col_names = results.columns.values.tolist()
        
        clf_col = ""
        for c in col_names:
            if "clf" in c and "bin" not in c:
                clf_col = c
        
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
