import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tanimoto_gp import TanimotoGP
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from xgboost import XGBRegressor
from glob import glob
from ffnn import FFNN


def rmse(truth, pred):
    return np.sqrt(mean_squared_error(truth, pred))


for filename in sorted(glob("data/A*.smi")):
    df = pd.read_csv(filename, names=["SMILES", "Name", "pIC50"], sep=" ")
    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    df['fp'] = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in df.mol]

    for _ in range(0, 10):
        train, test = train_test_split(df)
        X_train = np.asarray(list(train.fp.values))
        X_test = np.asarray(list(test.fp.values))
        y_train = train.pIC50.values
        y_test = test.pIC50.values

        tan_gp = TanimotoGP()
        tan_gp.fit(X_train, y_train)
        gp_pred, gp_var = tan_gp.predict(X_test)
        gp_r2 = r2_score(y_test, gp_pred)
        gp_rmse = rmse(y_test, gp_pred)

        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_rmse = rmse(y_test, xgb_pred)

        ff_nn = FFNN()
        ff_nn.fit(X_train, y_train)
        ff_pred = ff_nn.predict(X_test)
        ff_r2 = r2_score(y_test, ff_pred)
        ff_rmse = rmse(y_test, ff_pred)

        print(filename, gp_r2, xgb_r2, ff_r2, gp_rmse, xgb_rmse, ff_rmse)
