from abc import ABC
from typing import Any, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem #deprecate

NBITS = 1024
RADIUS = 3

class MorganFeaturizer:
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
