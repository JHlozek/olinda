"""Reference SMILES datamodule."""

from pathlib import Path
from typing import Any, Optional, Union
import shutil
import os

from cbor2 import dump
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds

from olinda.utils.utils import get_workspace_path, calculate_cbor_size


class ReferenceSmilesDM(pl.LightningDataModule):
    """Reference SMILES datamodule."""

    def __init__(
        self: "ReferenceSmilesDM",
        ref_df: pd.DataFrame,
        workspace: Union[str, Path] = None,
        batch_size: int = 32,
        num_workers: int = 1,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """Init.

        Args:
            workspace (Union[str, Path]): URLs or Path to the data files.
                Defaults to local workspace.
            batch_size (int): batch size. Defaults to 32.
            num_workers (int): workers to use. Defaults to 2.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
        """
        super().__init__()
        self.ref_df = ref_df
        self.workspace = workspace or get_workspace_path()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform = target_transform

    def prepare_data(self: "ReferenceSmilesDM") -> None:
        """Prepare data."""
        # Check if raw data files already present
        if os.path.exists(Path(self.workspace / "reference" / "reference_smiles.csv")) == False:
            self.ref_df.to_csv(Path(self.workspace / "reference" / "reference_smiles.csv"), header=False, index=False)
            
        # Check if whole processed data file already present
        lib_path = Path(self.workspace / "reference" / "reference_smiles.cbor")
        if lib_path.is_file() is False:
            # preprocess csv into a cbor file
            self.write_data(self.ref_df, lib_path)
        else:
            with open(lib_path, "rb") as stream:
                num_compounds = calculate_cbor_size(stream)
            if num_compounds != self.ref_df.shape[0]:
                self.write_data(self.ref_df, lib_path)

        self.setup_dataloaders()
            
    def write_data(self: "ReferenceSmilesDM", df: pd.DataFrame, output_path: str) -> None:
        # remove old dataloader if reference library is updated
        if os.path.exists(os.path.join(self.workspace, "reference", "reference_smiles_dl.joblib")):
            os.remove(os.path.join(self.workspace, "reference", "reference_smiles_dl.joblib"))

        with open(output_path, "wb") as stream:
                for i, row in tqdm(
                    df.iterrows(),
                    total=df.shape[0],
                    desc="Creating reference smiles dataset",
                ):
                    dump((i, str(row.to_list()[0])), stream)
        
    def setup_dataloaders(self: "ReferenceSmilesDM") -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
        """
        self.dataset_size = self.ref_df.shape[0]

        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(
                str(
                    (
                        self.workspace
                        / "reference"
                        / "reference_smiles.cbor"
                    ).absolute()
                )
            ),
            wds.cbors2_to_samples(),
            wds.batched(self.batch_size, partial=False),
        )

    def get_dataloader(self: "ReferenceSmilesDM") -> DataLoader:
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

        loader.length = (self.dataset_size * self.num_workers) // self.batch_size

        return loader

    def teardown(self: "ReferenceSmilesDM", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
