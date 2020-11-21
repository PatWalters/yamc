import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from ffnn import FFNN
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


filename = "A2a.smi"
df = pd.read_csv(filename, names=["SMILES", "Name", "pIC50"], sep=" ")
df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
df['fp'] = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in df.mol]
ff_nn = FFNN()

train, test = train_test_split(df)
X_train = np.asarray(list(train.fp.values))
y_train = train.pIC50.values
X_test = np.asarray(list(test.fp.values))
y_test = test.pIC50.values

ff_nn.fit(X_train, y_train)
pred = ff_nn.predict(X_test)
print(r2_score(y_test,pred))
