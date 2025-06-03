"""Featurizer for SMILES."""

from abc import ABC
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
import datamol as dm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem #deprecate

import torch

from olinda.utils.utils import get_package_root_path

NBITS = 2048
RADIUS = 3

class Featurizer(ABC):
    def featurize(self: "Featurizer", batch: Any) -> Any:
        """Featurize input batch.

        Args:
            batch (Any): batch of smiles

        Returns:
            Any: featurized outputs
        """
        pass

class MorganFeaturizer(Featurizer):
    def __init__(self: "MorganFeaturizer") -> None:
        self.name = "morganfeaturizer"
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS,fpSize=NBITS)

    def clip_sparse(self: "MorganFeaturizer", vect: List, nbits: int) -> List:
        l = [0] * nbits
        for i, v in vect.GetNonzeroElements().items():
            l[i] = v if v < 255 else 255
        return l

    def featurize(self: "MorganFeaturizer", batch: Any) -> Any:
        """Featurize input batch.

        Args:
            batch (Any): batch of smiles

        Returns:
            Any: featurized outputs
        """
        mols = [Chem.MolFromSmiles(smi) for smi in batch if smi is not None]
        ecfps = self.ecfp_counts(mols)
        return ecfps

    def ecfp_counts(self: "MorganFeaturizer", mols: List) -> List:
        """Create ECFPs from batch of smiles.

        Args:
            mols (List): batch of molecules

        Returns:
            List: batch of ECFPs
        """
        fps = [self.clip_sparse(self.mfpgen.GetCountFingerprint(mol), NBITS)
         if mol is not None else None for mol in mols
        ]
        return np.array(fps)   

class Flat2Grid(MorganFeaturizer):
    def __init__(self: "Flat2Grid") -> None:
        self.transformer = joblib.load(get_package_root_path() / "flat2grid.joblib")
        self.name = "flat2grid"

    def featurize(self: "Flat2Grid", batch: Any) -> Any:
        """Featurize input batch.

        Args:
            batch (Any): batch of smiles

        Returns:
            Any: featurized outputs
        """

        mols = [Chem.MolFromSmiles(smi) for smi in batch]
        ecfps = self.ecfp_counts(mols)
        return self.transformer.transform(ecfps)
        
    def ecfp_counts(self: "MorganFeaturizer", mols: List) -> List:
        """Create ECFPs from batch of smiles.

        Args:
            mols (List): batch of molecules

        Returns:
            List: batch of ECFPs
        """
        fps = [
            AllChem.GetMorganFingerprint(
                mol, radius=RADIUS, useCounts=True, useFeatures=True
            )
            for mol in mols
        ]
        nfp = np.zeros((len(fps), NBITS), np.uint8)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % NBITS
                nfp[i, nidx] += int(v)
        return nfp

class DatamolFeaturizer(Featurizer):
    def __init__(self: "DatamolFeaturizer") -> None:
        self.name = "datamolfeaturizer"
        
    def featurize(self: "DatamolFeaturizer", batch: Any) -> Any:
        """Featurize input batch.

        Args:
            batch (Any): batch of smiles

        Returns:
            Any: featurized outputs
        """
        mols = [dm.to_mol(smi) for smi in batch if smi is not None]
        descriptors = [dm.descriptors.compute_many_descriptors(mol) for mol in mols]
        X = np.array(pd.DataFrame(descriptors), dtype=np.float32)

        return X
