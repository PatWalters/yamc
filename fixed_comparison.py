#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from tanimoto_gp import TanimotoGP
from xgboost import XGBRegressor
from ffnn import FFNN


def rmse(truth, pred):
    return np.sqrt(mean_squared_error(truth, pred))


df = pd.read_csv("data/cv_splits.csv")
print("Generating Fingerprints", file=sys.stderr)
df['mol'] = [Chem.MolFromSmiles(x) for x in tqdm(df.SMILES, desc="Generating Molecules")]
df['fp'] = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in tqdm(df.mol, "Generating Fingerprints")]
print("Done", file=sys.stderr)

cols = df.columns
rnd_cols = [x for x in cols if x.startswith("RND")]
scaf_cols = [x for x in cols if x.startswith("SCAF")]
dataset_list = sorted(df.Dataset.unique())

res = []
for dataset in dataset_list:
    for col in rnd_cols + scaf_cols:
        train = df.query(f"Dataset == '{dataset}' and {col} == 'train'")
        test = df.query(f"Dataset == '{dataset}' and {col} == 'test'")

        X_train = np.asarray(list(train.fp.values))
        X_test = np.asarray(list(test.fp.values))
        y_train = train.pIC50.values
        y_test = test.pIC50.values

        tan_gp = TanimotoGP()
        tan_gp.fit(X_train, y_train)
        gp_pred, gp_var = tan_gp.predict(X_test)
        gp_r2 = r2_score(y_test, gp_pred)
        gp_rmse = rmse(y_test, gp_pred)
        print(gp_r2, gp_rmse)

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
        print([dataset, col, gp_r2, xgb_r2, ff_r2, gp_rmse, xgb_rmse, ff_rmse])
        sys.stdout.flush()
        res.append([dataset, col, gp_r2, xgb_r2, ff_r2, gp_rmse, xgb_rmse, ff_rmse])

res_df = pd.DataFrame(res, columns=["dataset", "split",'gp_r2', 'xgb_r2', 'ffnn_r2', 'gp_rmse', 'xgb_rmse', 'ffnn_rmse'])
res_df.to_csv("comparison.csv", index=False)