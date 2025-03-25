import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys

def smiles_to_ecfp(smiles, radius, n_bits):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return np.zeros((n_bits,))
  fpGen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
  fp = fpGen.GetFingerprint(mol)
  arr = np.zeros((1,))
  AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
  return arr

def smiles_to_maccs(smiles):
	mol = Chem.MolFromSmiles(smiles)

	if mol is not None:
		maccs = Chem.MACCSkeys.GenMACCSKeys(mol)
		maccs_array = list(maccs)
		return maccs_array
	else:
		return None