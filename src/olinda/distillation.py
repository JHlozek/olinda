"""Distillation module."""

from warnings import filterwarnings
filterwarnings(action="ignore")

import os
import glob
from pathlib import Path
from importlib import resources
import shutil
import math
import json
import tempfile
from typing import Any, Optional
from loguru import logger

from cbor2 import dump
import joblib
import pytorch_lightning as pl
import torch
from tqdm import tqdm

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import zipfile

import onnx
import pandas as pd

from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

from olinda.data import ReferenceSmilesDM, FeaturizedSmilesDM, GenericOutputDM
from olinda.featurizer import Featurizer, MorganFeaturizer
from olinda.generic_model import GenericModel
from olinda.models.distilled_model import DistilledModel
from olinda.tuner import ModelTuner, XgboostTuner
from olinda.utils.utils import calculate_cbor_size, get_workspace_path
from olinda.utils.s3 import download_s3_folder, ProgressPercentage
from olinda.reports.report import Reporter
from olinda.configs.vars import REF_FOLD_SIZE

### TODO: Improve object-oriented setup of distillation code segments
class Distiller(object):
    def __init__(self,
        featurizer: Optional[Featurizer] = MorganFeaturizer(),
        tuner: ModelTuner = XgboostTuner(),
        reference_smiles_dm: Optional[ReferenceSmilesDM] = None,
        featurized_smiles_dm: Optional[FeaturizedSmilesDM] = None,
        generic_output_dm: Optional[GenericOutputDM] = None,
        num_data: int = 100000,
        clean: bool = False,
        test: bool = False,
    ):
        """
        Args:
        featurizer (Optional[Featurizer]): Featurizer to use.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.
        tuner (ModelTuner): Tuner to use for selecting and optimizing student model.
        reference_smiles_dm (Optional[ReferenceSmilesDM]): Reference SMILES datamodules.
        featurized_smiles_dm (Optional[FeaturizedSmilesDM]): Reference Featurized SMILES datamodules.
        generic_output_dm (Optional[GenericOutputDM]): Precalculated training dataset for student model.
        test (bool): Run a test distillation on a smaller fraction of the dataset.
        """     
        self.working_dir = get_workspace_path()
        self.featurizer = featurizer
        self.tuner = tuner
        self.reference_smiles_dm = reference_smiles_dm
        self.featurized_smiles_dm = featurized_smiles_dm
        self.generic_output_dm = generic_output_dm
        self.num_data = num_data
        self.clean = clean
        if test:
            self.num_data = self.num_data // 10
        self.test = test


    def distill(self, model: Any, output_path: str = None) -> pl.LightningModule:
        """Distill models.
        
        Args:
            model (Any): Teacher Model.
            output_path (str): Path to save distilled onnx model.
        Returns:
            pl.LightningModule: Student Model.
        """
        
        if self.clean is True:
            clean_workspace(Path(self.working_dir), reference=True)
        
        # Convert model to a generic model
        model_path = model
        model = GenericModel(model)
        
        ref_library_path = os.path.join(resources.files("olinda"), "reference_library/olinda_reference_library.csv")
        ref_library_df = pd.read_csv(ref_library_path)
        self.reference_smiles_dm = ReferenceSmilesDM(ref_library_df, num_data=self.num_data)
        self.reference_smiles_dm.prepare_data()
        self.reference_smiles_dm.setup("train")
        if self.num_data > len(ref_library_df):
            self.num_data = len(ref_library_df)
        
        if model.type == "zairachem":
            #fetch_ref_library()            
            """
            zairachem_folds = math.ceil(self.num_data / REF_FOLD_SIZE) #Calculate number of folds required
            zaira_describe_path = os.path.join(model.name[len(model.type)+1:], "descriptors")
            with open(os.path.join(zaira_describe_path, "done_eos.json")) as used_desc_file:
                req_descs = json.load(used_desc_file)
            fetch_descriptors(zairachem_folds, req_descs)
            """
            self.featurized_smiles_dm = gen_featurized_smiles(self.reference_smiles_dm, self.featurizer, self.working_dir, num_data=self.num_data, clean=self.clean)
            self.featurized_smiles_dm.setup("train")
            student_training_dm = gen_model_output(model, self.featurized_smiles_dm, self.featurizer, ref_library_path, self.working_dir, self.num_data, self.clean)
        else:
            student_training_dm = self.generic_output_dm
                    
        # Select and Train student model
        student_model = self.tuner.fit(student_training_dm)
        model_onnx = convert_to_onnx(student_model, self.featurizer)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model_onnx.save(output_path)
        student_model.nn.save_model(output_path.split(".")[0] + ".json")

        r = Reporter(model_path, output_path, self.featurizer)
        r.report()

        return model_onnx

#Deprecated
def fetch_ref_library():
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket('olinda')
    
    path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors")
    os.makedirs(path, exist_ok=True)
    
    lib_name = "olinda_reference_library.csv"
    ref_lib = os.path.join(path, lib_name)
    # if no reference library or the size differs to the S3 bucket version
    if os.path.exists(ref_lib) == False or bucket.Object(key=lib_name).content_length != os.path.getsize(ref_lib):
        if os.path.exists(ref_lib):
            os.remove(ref_lib)
        bucket.download_file(
                "olinda_reference_library.csv", ref_lib,
                Callback=ProgressPercentage(bucket, "olinda_reference_library.csv")
                )

#Deprecated
def fetch_descriptors(
    num_folds: int,
    req_descs: list,
    ):
    """Check if required precalculated descriptor folds are on disk and fetch missing folds 
    
    Args:
        num_folds (int): Number of folds of 50k precalculated descriptors
    """
    
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket('olinda')
    
    local_path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors")
    
    for i in range(num_folds):
        fold = "olinda_reference_descriptors_" + str(i*50) + "_" + str((i+1)*50) + "k"
        if not os.path.exists(os.path.join(local_path, fold)):
            print("Downloading precalculated descriptors: fold " + str(i+1) + " of " + str(num_folds))
            # download compound lists and mapping
            s3_fold_data = os.path.join(fold, "data")
            local_fold_data = os.path.join(local_path, fold, "data")
            download_s3_folder("olinda", s3_fold_data, local_fold_data)
            bucket.download_file(os.path.join(fold, "reference_library.csv"), os.path.join(local_path, fold, "reference_library.csv"))
        
        # download raw descriptor files
        for desc in req_descs:
            s3_path = os.path.join(fold, "descriptors", desc)
            dest_path = os.path.join(local_path, fold, "descriptors", desc)
        
            if not os.path.exists(dest_path):
                download_s3_folder("olinda", s3_path, dest_path)
                assert os.path.exists(dest_path)

def gen_featurized_smiles(
    reference_smiles_dm: pl.LightningDataModule,
    featurizer: Featurizer,
    working_dir: Path,
    num_data,
    clean: bool = False,
) -> pl.LightningDataModule:
    """Generate featurized smiles representation dataset.

    Args:
        reference_smiles_dm (pl.LightningDataModule): Reference SMILES datamodule.
        featurizer (Featurizer): Featurizer to use.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.

    Returns:
        pl.LightningDataModule: Dateset with featurized smiles.
    """

    if clean is True:
        clean_workspace(Path(working_dir), featurizer=featurizer)
        reference_smiles_dl = reference_smiles_dm.train_dataloader()
    else:
        try:
            reference_smiles_dl = joblib.load(
                Path(working_dir / "reference" / "reference_smiles_dl.joblib")
            )
        except Exception:
            reference_smiles_dl = reference_smiles_dm.train_dataloader()

    # Save dataloader for resuming
    joblib.dump(
        reference_smiles_dl,
        Path(working_dir) / "reference" / "reference_smiles_dl.joblib",
    )

    # find existing featurization file and calculate stop_step
    try:
        with open(
            Path(working_dir)
            / "reference"
            / f"featurized_smiles_{(type(featurizer).__name__.lower())}.cbor",
            "rb",
        ) as feature_stream:
            stop_step = calculate_cbor_size(feature_stream)
    except Exception:
        stop_step = 0

    with open(
        Path(working_dir)
        / "reference"
        / f"featurized_smiles_{(type(featurizer).__name__.lower())}.cbor",
        "ab",
    ) as feature_stream:
        for i, batch in tqdm(
            enumerate(iter(reference_smiles_dl)),
            total=reference_smiles_dl.length,
            desc="Featurizing",
        ):  
            if i < stop_step // len(batch[0]):
                continue   
            if i >= reference_smiles_dl.length:
                break
            if i >= num_data:
                break
                     
            output = featurizer.featurize(batch[1])
            for j, elem in enumerate(batch[0]):
                dump((elem.tolist(), batch[1][j], output[j].tolist()), feature_stream)
    
    featurized_smiles_dm = FeaturizedSmilesDM(Path(working_dir), featurizer, num_data=num_data)
    
    return featurized_smiles_dm

def gen_model_output(
    model: GenericModel,
    featurized_smiles_dm: pl.LightningDataModule,
    featurizer: Featurizer,
    ref_library_path: str,
    working_dir: Path,
    num_data: int,
    clean: bool = False,
) -> pl.LightningDataModule:

    """Generate featurized smiles representation dataset.

    Args:
        featurized_smiles_dm ( pl.LightningDataModule): Featurized SMILES to use as inputs.
        model (GenericModel): Wrapped Teacher model.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.

    Returns:
        pl.LightningDataModule: Dateset with featurized smiles.
    """
    if issubclass(type(featurizer), MorganFeaturizer):
        feat_name = "morgan"
    feat_dl_path = os.path.join(working_dir, model.name, "featurized_smiles_dl_" + feat_name + ".joblib")
    os.makedirs(os.path.join(working_dir, model.name), exist_ok=True)

    if clean is True:
        featurized_smiles_dl = featurized_smiles_dm.train_dataloader()
    else:
        try:
            featurized_smiles_dl = joblib.load(feat_dl_path)
        except Exception:
            featurized_smiles_dl = featurized_smiles_dm.train_dataloader()

    # Save dataloader for resuming
    joblib.dump(featurized_smiles_dl, feat_dl_path)

    with open(
        Path(working_dir) / (model.name) / "model_output.cbor", "wb"
    ) as output_stream:
    
        if model.type == "zairachem":
            # correct wrong zairachem predictions before training Olinda
            training_output = model.get_training_preds()
            for i, row in enumerate(training_output.iterrows()):
                if row[1]["pred"] >= 0.0 and row[1]["pred"] <= 0.5 and row[1]["true"] == 1.0:
                    training_output.at[i, "pred"] = 1.0 - row[1]["pred"]
                elif row[1]["pred"] >= 0.5 and row[1]["pred"] <= 1.0 and row[1]["true"] == 0:
                    training_output.at[i, "pred"] = 1.0 - row[1]["pred"]
            
            ref_compounds_needed = num_data - training_output.shape[0]
            if ref_compounds_needed < 0:
                ref_compounds_needed = 0
            zaira_distill_path = os.path.join(model.name[len(model.type)+1:], "distill") #get model root
            output = gen_zaira_preds(model, ref_library_path, zaira_distill_path, ref_compounds_needed)
        
            """
            # weight by data source: training/reference
            # inverse of proportion of training compounds to all compounds
            train_weight = 1 #round((training_output.shape[0] + num_data) / len(training_output), 2)
            """
            
            # inverse of ratio of predicted active to inactive 
            y_bin_train = [1 if val > 0.5 else 0 for val in training_output["pred"]]
            y_bin_ref = [1 if val > 0.5 else 0 for val in output["pred"]]
            active_weight = 1 #(y_bin_train.count(0) + y_bin_ref.count(0)) / (y_bin_train.count(1) + y_bin_ref.count(1)) #inactive to active ratio
            
            
            print("Creating model prediction files")
            output_list = []
            train_counter = 0
            for i, row in training_output.dropna().iterrows():
                fp = featurizer.featurize([row["smiles"]])
                if fp is None:
                    continue
                if row["pred"] > 0.5:
                    weight = min(active_weight, 100) #prevent extreme weighting
                else:
                    weight = max(1/active_weight, 0.01)
                
                output_list.append([row["smiles"], row["pred"], weight])
                train_counter += 1
                dump((i, row["smiles"], fp[0].tolist(), [row["pred"]], [weight]), output_stream)
            output_df = pd.DataFrame(output_list, columns=["smiles", "prediction", "weight"])
            output_df.to_csv(os.path.join(zaira_distill_path, "original_training_set.csv"), index=False)
            
        ref_counter = 0        
        for i, batch in tqdm(
            enumerate(iter(featurized_smiles_dl)),
            total=featurized_smiles_dl.length,
            desc="Creating model output",
        ):
            if model.type == "zairachem":
                # final dataset a multiple of batch
                combined_count = train_counter + featurized_smiles_dl.length*len(batch[0])
                target_count = combined_count // len(batch[0]) * len(batch[0]) 
                for j, elem in enumerate(batch[1]):
                    if ref_counter + train_counter == target_count:
                        break
                    if not output[output["smiles"] == elem].empty:
                        pred_val = output[output["smiles"] == elem]["pred"].iloc[0]
                        if pred_val > 0.5:
                            weight = active_weight
                        else:
                            weight = 1/active_weight
                        output_list.append([elem, pred_val, weight])
                        ref_counter += 1
                        dump((j, elem, batch[2][j], [pred_val], [weight]), output_stream)   
             
            elif model.type == "ersilia":
                output = model(batch[1])
                for j, elem in enumerate(batch[1]):
                    dump((j, elem, batch[2][j], [output[j].tolist()]), output_stream)

            else:
            	output = model(torch.tensor(batch[2]))
            	for j, elem in enumerate(batch[1]):
            	    dump((j, elem, batch[2][j], output[j].tolist()), output_stream)

    if model.type == "zairachem":
        output_df = pd.DataFrame(output_list, columns=["smiles", "prediction", "weight"])
        output_df.to_csv(os.path.join(zaira_distill_path, "full_training_set.csv"), index=False)
        model_output_dm = GenericOutputDM(Path(working_dir / (model.name)), zaira_training_size = training_output.shape[0])
    else:
        model_output_dm = GenericOutputDM(Path(working_dir / (model.name)))   
        
    return model_output_dm


def gen_zaira_preds(
    model: GenericModel,
    ref_library_path: str,
    zaira_distill_path: str,
    ref_size,
) -> pl.LightningDataModule:

    """Generate featurized smiles representation dataset.

    Args:
        featurized_smiles_dm ( pl.LightningDataModule): Featurized SMILES to use as inputs.
        model (GenericModel): Wrapped Teacher model.
        ref_library_dir (Path): Path to reference compounds
        ref_size (int): Number of predictions to calculate

    Returns:
        pl.LightningDataModule: Dateset with featurized smiles.
    """
    output = None
    if os.path.exists(os.path.join(zaira_distill_path, "reference_library_predictions.csv")): #check if predictions already calculated
        output = pd.read_csv(os.path.join(zaira_distill_path, "reference_library_predictions.csv"))            
    else:
        if len(glob.glob(os.path.join(zaira_distill_path, "*"))) > 0: # clear dir if distillation did not finish correctly
            shutil.rmtree(zaira_distill_path)
            os.mkdir(zaira_distill_path)
        output = pd.DataFrame(columns = ["smiles", 'pred'])

        for i in range(math.ceil(ref_size/REF_FOLD_SIZE)):
           logger.info("Getting ZairaChem predictions for fold " + str(i+1) + " of " + str(math.ceil(ref_size/REF_FOLD_SIZE)))
           #folder = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "olinda_reference_descriptors_" + str(i*50) + "_" + str((i+1)*50) + "k")
           #smiles_input_path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", folder, "reference_library.csv")
           preds = model(ref_library_path)
           output = pd.concat([output, preds])    
        output = output[["smiles", "pred"]]   
         
        # save to zairachem model folder
        output.to_csv(os.path.join(zaira_distill_path, "reference_library_predictions.csv"), index=False)
    return output


def convert_to_onnx(
    model: pl.LightningModule,
    featurizer: Featurizer,
) -> onnx.onnx_ml_pb2.ModelProto:
    """Convert student model to ONNX format

    Args:
        model (GenericModel): Wrapped Student model.
        featurizer (Featurizer): Featurizer to test data shape.

    Returns:
        onnx.onnx_ml_pb2.ModelProto: ONNX formatted model
    """
    
    test_desc = featurizer.featurize(["CCC"])
    model_onnx = convert_xgboost(model.nn, 'tree-based classifier',
                             [('input', FloatTensorType([None, test_desc.shape[1]]))])

    model_onnx = DistilledModel(model_onnx)
    return model_onnx  

def clean_workspace(
    working_dir: Path, featurizer: Featurizer = None, reference: bool = False
) -> None:
    """Clean workspace.

    Args:
        working_dir (Path): Path of the working directory.
        model (GenericModel): Wrapped Teacher model.
        featurizer (Featurizer): Featurizer to use.
    """
    
    curr_ref_smiles_path = Path(working_dir) / "reference" / "reference_smiles.csv"
    orig_ref_smiles_path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "olinda_reference_library.csv")
    
    if featurizer:
        if os.path.exists(os.path.join(working_dir, "reference", "reference_smiles_dl.joblib")):
            os.remove(os.path.join(working_dir, "reference", "reference_smiles_dl.joblib"))
        if os.path.exists(os.path.join(working_dir, "reference", "featurized_smiles_" + type(featurizer).__name__.lower() + ".cbor")):
            os.remove(os.path.join(working_dir, "reference", "featurized_smiles_" + type(featurizer).__name__.lower() + ".cbor"))
    
    if reference and os.path.exists(curr_ref_smiles_path):
        if os.path.exists(orig_ref_smiles_path):
            curr_df = pd.read_csv(curr_ref_smiles_path, header=None, names=["SMILES"])
            orig_df = pd.read_csv(orig_ref_smiles_path)
            if not curr_df.equals(orig_df):
                shutil.rmtree(Path(working_dir) / "reference")
        else:
            shutil.rmtree(Path(working_dir) / "reference")

