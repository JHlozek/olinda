"""Featurized SMILES datamodule."""

from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import webdataset as wds

from olinda.featurizer import Featurizer, MorganFeaturizer
from olinda.utils.utils import calculate_cbor_size


class FeaturizedSmilesDM(pl.LightningDataModule):
    """Featurized SMILES datamodule."""

    def __init__(
        self: "FeaturizedSmilesDM",
        workspace_dir: Union[str, Path],
        featurizer: Featurizer = MorganFeaturizer(),
        num_data: int = 100000,
        batch_size: int = 32,
        num_workers: int = 1,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """Init.

        Args:
            workspace_dir (Union[str, Path]): Path to the data files.
            featurizer (Featurizer): Featurizer to use.
            batch_size (int): batch size. Defaults to 32.
            num_workers (int): workers to use. Defaults to 2.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
        """
        super().__init__()
        self.workspace_dir = workspace_dir
        self.featurizer = featurizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform = target_transform
        self.num_data = num_data
        

    def setup(self: "FeaturizedSmilesDM", stage: Optional[str]) -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state

        Raises:
            Exception : Data file not available.

        """
        # Check if data files are available
        file_path = (
            Path(self.workspace_dir)
            / "reference"
            / f"featurized_smiles_{(type(self.featurizer).__name__.lower())}.cbor"
        )
        if file_path.is_file() is not True:
            raise Exception(f"Data file not available at {file_path.absolute()}")

        if stage == "train":
            self.train_dataset_size = self.num_data
        elif stage == "val":
            self.val_dataset_size = self.num_data // 10

        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(str(file_path.absolute())),
            wds.cbors2_to_samples(),
            wds.batched(self.batch_size, partial=False),
        )

    def train_dataloader(self: "FeaturizedSmilesDM") -> DataLoader:
        """Train dataloader.

        Returns:
            DataLoader: train dataloader
        """
        loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

        loader.length = (self.train_dataset_size * self.num_workers) // self.batch_size

        return loader

    def val_dataloader(self: "FeaturizedSmilesDM") -> DataLoader:
        """Val dataloader.

        Returns:
            DataLoader: val dataloader
        """
        loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

        loader.length = (self.val_dataset_size * self.num_workers) // self.batch_size

        return loader

    def teardown(self: "FeaturizedSmilesDM", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
