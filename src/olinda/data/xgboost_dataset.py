"""A wapper to provide a tensorflow dataset interface."""

import numpy as np

from olinda.data import GenericOutputDM


class XgboostDataset:
    """A wapper to provide a tensorflow dataset interface."""

    def __init__(
        self,
        datamodule: GenericOutputDM,
        stage: str = "train",
        only_X: bool = True,
        only_Y: bool = True,
        weights: bool = True,
        smaller_set: bool = False,
    ):
        self.only_X = only_X
        self.only_Y = only_Y
        self.weights = weights
        self.smaller_set = smaller_set

        datamodule.setup(stage, only_X, only_Y, weights=True, batched=False, smaller_set=self.smaller_set)
        self.dataset = datamodule.dataset

        if stage == "train":
            self.loader = datamodule.train_dataloader()
        elif stage == "val":
            self.loader = datamodule.val_dataloader()

        self.setup()

    def setup(self):
        self.X, self.y, self.weights = [], [], []
        for sample in self.loader:
            self.X.append(sample[0])
            self.y.append(sample[1][0])
            self.weights.append(sample[2][0])
        

