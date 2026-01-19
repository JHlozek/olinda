#Script to calculate new folds of ZairaChem descriptors to be stored in AWS S3

import os
import csv
import pandas as pd
import numpy as np
import shutil
import h5py

from ersilia import ErsiliaModel

DATA_SUBFOLDER = "data"

class DescriptorCalculator():
    def __init__(self, smiles_csv, output_path):
        self.smiles_path = smiles_csv
        self.output_path = output_path
        os.makedirs(os.path.join(self.output_path, "descriptors"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "data"), exist_ok=True)
        self.data_path = os.path.join(self.output_path, "data", "data.csv")
        
    def calculate(self):
        # prepare input files and filter out problem compounds from grover calculations
        self._data_files()
        
        self.df = pd.read_csv(os.path.join(self.output_path, "reference_library.csv"))
        self.smiles_list = self.df["SMILES"].to_list()
        
        # zairachem raw descriptors ersilia api
        base_desc = ["eos4u6p", "eos5axz", "eos7jio", "eos78ao", "eos3cf4", "eos2gw4", "eos7w6n"] 
        for desc in base_desc:
            print(desc)
            path = os.path.join(self.output_path, "descriptors", desc)
            os.makedirs(path, exist_ok=True)
            with ErsiliaModel(desc) as em:
                em.api(input=self.data_path, output=os.path.join(path, "raw.h5"))
        
    def _data_files(self):
        raw_df = pd.read_csv(self.smiles_path)
        indx_list = [i for i, smi in enumerate(raw_df["SMILES"].to_list())]
        cmpd_list = ["CID" + str(i).zfill(4) for i in indx_list]
        
        raw_df.rename(columns = {"SMILES":"smiles"}, inplace=True)
        raw_df["compound_id"] = cmpd_list
        raw_df.to_csv(os.path.join(self.output_path, "data", "compounds.csv"), index=False)
        
        mapping = pd.DataFrame(list(zip(indx_list, indx_list, cmpd_list)), columns=["orig_idx", "uniq_idx", "compound_id"])    
        mapping.to_csv(os.path.join(self.output_path, "data", "mapping.csv"))
        
        self._screen_smiles()
        
        self.df = pd.read_csv(os.path.join(self.output_path, "data", "data.csv"))
        self.df.rename(columns = {"smiles":"SMILES"}, inplace=True)
        self.df[["SMILES"]].to_csv(os.path.join(self.output_path, "reference_library.csv"), index=False)
        
    def _screen_smiles(self):
        print("Check SMILES with Grover")
        raw_smiles_path = os.path.join(self.output_path, "descriptors", "eos7w6n_initial.h5")
        with ErsiliaModel("grover-embedding") as em:
                em.api(input=self.smiles_path, output=raw_smiles_path)
        
        with h5py.File(raw_smiles_path, "r") as data_file:
            keys = data_file["Keys"][:]
            inputs = data_file["Inputs"][:]
            features = data_file["Features"][:]
            values = data_file["Values"][:]
        
        drop_indxs = [i for i, row in enumerate(np.isnan(values)) if True in row]    
        
        # filter out problematic smiles from data files
        smiles_strings = [smi.decode("utf-8") for smi in np.delete(inputs, drop_indxs)]    
        indx_list = [i for i, smi in enumerate(smiles_strings)]
        cmpd_list = ["CID" + str(i).zfill(4) for i in indx_list]
        
        df = pd.DataFrame(list(zip(cmpd_list, smiles_strings)), columns=["compound_id", "smiles"])
        df.to_csv(self.data_path, index=False)
        
        mapping = pd.DataFrame(list(zip(indx_list, indx_list, cmpd_list)), columns=["orig_idx", "uniq_idx", "compound_id"])    
        mapping.to_csv(os.path.join(self.output_path, "data", "mapping.csv"))
        
        os.remove(raw_smiles_path)

