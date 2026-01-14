"""A wrapper to standardize models."""

from typing import Any

import pytorch_lightning as pl
import torch.nn as nn

import onnx
import os
import pickle

from olinda.models.base import DistillBaseModel
import xgboost as xgb

class GenericModel(DistillBaseModel):
    def __init__(self: "GenericModel", model: Any) -> None:
        """Init.

        Args:
            model (Any): Any ML model

        Raises:
            Exception : Unsupported Model
        """
        super().__init__()
        # Check type of model and convert accordingly
        if issubclass(type(model), (pl.LightningModule, nn.Module)):
            self.nn = model
            self.type = "pytorch"
            self.name = type(model).__name__.lower()

        elif issubclass(type(model), (xgb.core.Booster)):
            self.nn = model
            self.type = "xgboost"
            self.name = type(model).__name__.lower()
      
        elif type(model) is str:
            if model[:3] == "eos":
            	from olinda.utils.ersilia.utils import run_ersilia_api_in_context
            	self.nn = run_ersilia_api_in_context(model)
            	self.type = "ersilia"
            	self.name = self.type + "_" + model
            else:
            	from olinda.utils.zairachem.utils import run_zairachem, get_zairachem_training_preds
            	self.nn = run_zairachem(model)
            	self.type = "zairachem"
            	self.name = self.type + "_" + model
            	self.get_training_preds = get_zairachem_training_preds(model)

        else:
            raise Exception(f"Unsupported Model type: {type(model)}")

    def forward(self: "GenericModel", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): Input

        Returns:
            Any: Ouput
        """
        return self.nn(x)