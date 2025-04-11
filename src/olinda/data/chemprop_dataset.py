"""A wapper to provide a tensorflow dataset interface."""

import torch
import numpy as np
from chemprop.data import MoleculeDataset, MoleculeDatapoint
#from chemprop.featurizers.molecule import MorganCountFeaturizer
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from olinda.data import GenericOutputDM


class ChempropDataset:
    """A wapper to provide a tensorflow dataset interface."""

    def __init__(
        self,
        datamodule: GenericOutputDM,
        stage: str,
        only_X: bool = True,
        only_Y: bool = True,
        weights: bool = False,
        smaller_set: bool = False,
    ):
        self.only_X = only_X
        self.only_Y = only_Y
        self.weights = weights
        self.smaller_set = smaller_set

        datamodule.setup(stage, only_X, only_Y, weights, smiles=True, batched=False, smaller_set=self.smaller_set)
        self.dataset = datamodule.dataset

        if stage == "train":
            self.loader = datamodule.train_dataloader()
        elif stage == "val":
            self.loader = datamodule.val_dataloader()

        self.setup()
        
    def setup(self):
        self.all_data = [MoleculeDatapoint.from_smi(datum[0], datum[1][0]) for datum in iter(self.loader)]
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        self.dataset = MoleculeDataset(self.all_data, self.featurizer)
